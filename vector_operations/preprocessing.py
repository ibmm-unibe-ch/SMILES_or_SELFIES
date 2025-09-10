"""Preprocessing pipeline for SMILES and SELFIES molecular representations."""

import logging
import multiprocessing
import os
import pickle
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import selfies
from constants import DESCRIPTORS, HEADER, PROCESSED_PATH, PROJECT_PATH, OTHERS
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.Descriptors import Chi0v, Kappa1, MolLogP, MolMR
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from tqdm import tqdm


def check_valid(input_str: str) -> bool:
    """Check validity of SMILES string.
    
    Args:
        input_str: SMILES string to validate
        
    Returns:
        True if valid, False if invalid
    """
    return Chem.MolFromSmiles(input_str) is not None

def canonize_smile(input_str: str, remove_identities: bool = True) -> Optional[str]:
    """Canonize SMILES string.
    
    Args:
        input_str: SMILES input string
        remove_identities: Whether to remove atom mapping numbers
        
    Returns:
        Canonical SMILES string or None if invalid
    """
    mol = Chem.MolFromSmiles(input_str)
    if mol is None:
        return None
    if remove_identities:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def check_canonized(input_str: str) -> bool:
    """Check if a SMILES string is already canonized.
    
    Args:
        input_str: SMILES string to check
        
    Returns:
        True if already in canonical form
    """
    return canonize_smile(input_str) == input_str

def translate_selfie(smile: str) -> Tuple[Optional[str], int]:
    """Translate SMILES to SELFIES.
    
    Args:
        smile: SMILES string to convert
        
    Returns:
        Tuple of (SELFIES string, length) or (None, -1) on error
    """
    try:
        selfie = selfies.encoder(smile)
        return (selfie, selfies.len_selfies(selfie))
    except Exception as e:
        logging.error(f"SELFIES translation error: {e}")
        return (None, -1)

def translate_smile(selfie: str) -> Optional[str]:
    """Translate SELFIES to SMILES.
    
    Args:
        selfie: SELFIES string to convert
        
    Returns:
        Canonical SMILES or None on error
    """
    try:
        smile = selfies.decoder(selfie)
        return canonize_smile(smile)
    except Exception as e:
        logging.error(f"SMILES translation error: {e}")
        return None

class MolecularPreprocessor:
    """Handles preprocessing of molecular data from SMILES/SELFIES representations."""
    def __init__(self):
        self.calculator = MolecularDescriptorCalculator(DESCRIPTORS)
        self.logger = logging.getLogger(__name__)
    
    
    def calc_descriptors(self, mol_string: str) -> Dict[str, Optional[float]]:
        """Calculate molecular descriptors.
        
        Args:
            mol_string: SMILES string of molecule
            
        Returns:
            Dictionary of descriptor names to values
        """
        mol = Chem.MolFromSmiles(mol_string)
        if mol is None:
            return {key: None for key in DESCRIPTORS}
        calcs = self.calculator.CalcDescriptors(mol)
        return dict(zip(DESCRIPTORS, calcs))
    
    def calc_other_descriptors(mol_string: str) -> Dict[str, Optional[float]]:
        """Calculate additional molecular descriptors.
        
        Args:
            mol_string: SMILES string of molecule
            
        Returns:
            Dictionary of descriptor names to values
        """
        mol = Chem.MolFromSmiles(mol_string)
        if mol is None:
            return {key: None for key in OTHERS}
        return {
            "Chi0v": Chi0v(mol),
            "Kappa1": Kappa1(mol),
            "MolLogP": MolLogP(mol),
            "MolMR": MolMR(mol),
            "QED": Chem.QED.qed(mol),
        }
    
    def process_mol(self, mol: str) -> Tuple[Optional[List], str]:
        """Process a single molecule given as SMILES string.
        
        Args:
            mol: SMILES string of molecule
            
        Returns:
            Tuple of (processed data, validity status)
        """
        if not check_valid(mol):
            self.logger.warning(f"Invalid SMILES: {mol}")
            return None, "invalid_smile"
            
        canon = canonize_smile(mol)
        descriptors = self.calc_descriptors(canon)
        selfie, selfie_length = translate_selfie(canon)
        
        if selfie_length == -1:
            return None, "invalid_selfie"
            
        descriptors["SELFIE"] = selfie
        descriptors["SELFIE_LENGTH"] = selfie_length
        descriptors["SMILE"] = canon
        
        return list(descriptors.values()), "valid"
    
    def process_mol_other(self, mol: str) -> Tuple[Optional[Dict], str]:
        """Process a molecule for additional descriptors.
        
        Args:
            mol: SMILES string of molecule
            
        Returns:
            Tuple of (processed data, validity status)
        """
        if not check_valid(mol):
            self.logger.warning(f"Invalid SMILES: {mol}")
            return None, "invalid_smile"
            
        canon = self.canonize_smile(mol)
        descriptors = self.calc_other_descriptors(canon)
        descriptors["SMILES"] = canon
        
        return descriptors, "valid"
    
    def create_isomers(self, mol_string: str, isomers: int = 0) -> Tuple[List[Dict], Dict]:
        """Create stereoisomers for a molecule.
        
        Args:
            mol_string: SMILES string of base molecule
            isomers: Number of isomers to generate
            
        Returns:
            Tuple of (list of processed isomers, statistics)
        """
        if isomers <= 0:
            return None, {}
            
        mol = Chem.MolFromSmiles(mol_string)
        opts = StereoEnumerationOptions(maxIsomers=isomers, rand=mol_string)
        isomers = [Chem.MolToSmiles(isomer) for isomer in EnumerateStereoisomers(mol, opts)]
        
        output = []
        statistics = {}
        
        for isomer in isomers:
            processed, validity = self.process_mol(isomer)
            if processed is not None:
                output.append(processed)
            statistics[f"{validity}_isomer"] = statistics.get(f"{validity}_isomer", 0) + 1
            
        return output, statistics
    
    def process_mol_file(self, input_file: Path, isomers: int = 0) -> Tuple[List[Dict], Dict]:
        """Process a file containing multiple molecules.
        
        Args:
            input_file: Path to input file
            isomers: Number of isomers to generate per molecule
            
        Returns:
            Tuple of (processed molecules, statistics)
        """
        statistics = {}
        output = []
        
        with open(input_file, "r") as f:
            smiles = f.readlines()
            
        for smile in tqdm(smiles):
            processed, validity = self.process_mol(smile)
            if processed is not None:
                processed = [processed]
                processed_isomers, isomer_stats = self.create_isomers(smile, isomers)
                if processed_isomers:
                    processed.extend(processed_isomers)
                output.extend(processed)
                stats = {validity: 1} | isomer_stats
                
                for key, item in stats.items():
                    statistics[key] = statistics.get(key, 0) + item
                    
        return output, statistics
    
    def process_mol_file_other(self, input_file: Path) -> List[Dict]:
        """Process a file for additional descriptors.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of processed molecules
        """
        output = []
        with open(input_file, "r") as f:
            smiles = f.readlines()
            
        for smile in tqdm(smiles):
            processed, _ = self.process_mol_other(smile)
            if processed is not None:
                output.append(processed)
                
        return output
    
    def process_mol_files(self, input_folder: Path, output_dir: Path, isomers: int = 0) -> Tuple[List[Path], Dict]:
        """Process all files in a directory.
        
        Args:
            input_folder: Directory containing input files
            output_dir: Directory to save processed files
            isomers: Number of isomers to generate per molecule
            
        Returns:
            Tuple of (list of output paths, statistics)
        """
        paths = []
        statistics = {}
        input_files = list(input_folder.glob("*"))
        
        with multiprocessing.Pool(len(input_files)) as pool:
            results = pool.starmap(
                self.process_mol_file, 
                [(f, isomers) for f in input_files]
            )
            
        for result in results:
            curr_output, curr_statistics = result
            curr_path = output_dir / "".join(
                c for c in str(uuid.uuid4()) if c.isalnum()
            )
            
            pd.DataFrame(curr_output).to_csv(curr_path, header=HEADER, index=False)
            paths.append(curr_path)
            
            for key, item in curr_statistics.items():
                statistics[key] = statistics.get(key, 0) + item
                
        return paths, statistics
    
    def process_mol_files_other(self, input_folder: Path, output_dir: Path) -> List[Path]:
        """Process all files for additional descriptors.
        
        Args:
            input_folder: Directory containing input files
            output_dir: Directory to save processed files
            
        Returns:
            List of output paths
        """
        paths = []
        input_files = list(input_folder.glob("*"))
        
        with multiprocessing.Pool(len(input_files)) as pool:
            results = pool.map(self.process_mol_file_other, input_files)
            
        for result in results:
            curr_path = output_dir / "".join(
                c for c in str(uuid.uuid4()) if c.isalnum()
            )
            pd.DataFrame(result).to_csv(curr_path, index=False)
            paths.append(curr_path)
            
        return paths
    
    def merge_dataframes(paths_path: Path, output_path: Path, remove_paths: bool = False) -> pd.DataFrame:
        """Merge multiple dataframes.
        
        Args:
            paths_path: Path to pickle file containing dataframe paths
            output_path: Path to save merged dataframe
            remove_paths: Whether to delete intermediate files
            
        Returns:
            Merged DataFrame
        """
        with open(paths_path, "rb") as f:
            paths = pickle.load(f)
            
        logging.info(f"Merging {len(paths)} dataframes")
        merged = pd.concat([pd.read_csv(p) for p in paths])
        merged.to_csv(output_path, index=False)
        
        if remove_paths:
            for p in paths:
                p.unlink()
            paths_path.unlink()
            
        return merged
    
    def check_duplicates(df: pd.DataFrame, duplicated_rows_column: str = "SMILES") -> pd.DataFrame:
        """Remove duplicate rows based on a column.
        
        Args:
            df: DataFrame to process
            duplicated_rows_column: Column to check for duplicates
            
        Returns:
            Deduplicated DataFrame
        """
        duplicates = df.duplicated(subset=[duplicated_rows_column])
        dup_count = duplicates.sum()
        
        logging.info(f"Found {dup_count} duplicates in column '{duplicated_rows_column}'")
        return df[~duplicates]
    
    def run_standard_pipeline(self, output_dir: Path, isomers: int = 0):
        """Run the standard processing pipeline.
        
        Args:
            output_dir: Directory to save outputs
            isomers: Number of isomers to generate
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths, stats = self.process_mol_files(
            PROJECT_PATH / "download_10m",
            output_dir,
            isomers
        )
        
        self._log_processing_stats(stats)
        
        paths_file = output_dir / "paths.pickle"
        with open(paths_file, "wb") as f:
            pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        merged_path = output_dir / "full_10m.csv"
        merged = self.merge_dataframes(paths_file, merged_path, True)
        
        dedup_path = output_dir / "full_deduplicated_isomers.csv"
        self.check_duplicates(merged).to_csv(dedup_path, header=HEADER, index=False)
    
    def run_other_descriptors_pipeline(self, output_dir: Path):
        """Run pipeline for additional descriptors.
        
        Args:
            output_dir: Directory to save outputs
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = self.process_mol_files_other(
            PROJECT_PATH / "download_10m",
            output_dir
        )
        
        paths_file = output_dir / "paths.pickle"
        with open(paths_file, "wb") as f:
            pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        merged_path = output_dir / "full_10m.csv"
        merged = self.merge_dataframes(paths_file, merged_path, True)
        
        dedup_path = output_dir / "full_deduplicated_standard.csv"
        self.check_duplicates(merged).to_csv(dedup_path, index=False)
    
    def _log_processing_stats(self, stats: Dict[str, int]):
        """Log processing statistics.
        
        Args:
            stats: Dictionary of processing statistics
        """
        invalid_smile = stats.get("invalid_smile", 0)
        invalid_selfie = stats.get("invalid_selfie", 0)
        valid = stats.get("valid", 0)
        
        total = invalid_smile + invalid_selfie + valid
        passed_smile = total - invalid_smile
        
        self._log_stage_stats(
            "SMILES", invalid_smile, passed_smile, total
        )
        
        passed_selfie = passed_smile - invalid_selfie
        self._log_stage_stats(
            "SELFIES", invalid_selfie, passed_selfie, passed_smile
        )
        
        # Log isomer statistics if present
        if any("isomer" in k for k in stats):
            self._log_isomer_stats(stats)
    
    def _log_stage_stats(self, stage: str, invalid: int, passed: int, total: int):
        """Log statistics for a processing stage.
        
        Args:
            stage: Name of processing stage
            invalid: Number of invalid items
            passed: Number of passed items
            total: Total number of items
        """
        if total == 0:
            return
            
        self.logger.info(
            f"{stage} stage: {invalid} invalid, {passed} passed "
            f"({invalid/total*100:.2f}% invalid)"
        )
    
    def _log_isomer_stats(self, stats: Dict[str, int]):
        """Log isomer-specific statistics.
        
        Args:
            stats: Dictionary containing isomer statistics
        """
        invalid_smile_iso = stats.get("invalid_smile_isomer", 0)
        invalid_selfie_iso = stats.get("invalid_selfie_isomer", 0)
        valid_iso = stats.get("valid_isomer", 0)
        
        total_iso = invalid_smile_iso + invalid_selfie_iso + valid_iso
        passed_smile_iso = total_iso - invalid_smile_iso
        
        self._log_stage_stats(
            "Isomer SMILES", invalid_smile_iso, passed_smile_iso, total_iso
        )
        
        passed_selfie_iso = passed_smile_iso - invalid_selfie_iso
        self._log_stage_stats(
            "Isomer SELFIES", invalid_selfie_iso, passed_selfie_iso, passed_smile_iso
        )


def main():
    """Run the complete preprocessing pipeline."""
    preprocessor = MolecularPreprocessor()
    
    # Standard processing
    standard_dir = PROCESSED_PATH / "standard"
    preprocessor.run_standard_pipeline(standard_dir, isomers=0)

    # Isomer processing
    isomers_dir = PROCESSED_PATH / "isomers"
    preprocessor.run_standard_pipeline(isomers_dir, isomers=1)
    
    # Additional descriptors processing
    other_dir = PROCESSED_PATH / "standard_other"
    preprocessor.run_other_descriptors_pipeline(other_dir)
    
    # Merge results
    main_path = standard_dir / "full_deduplicated_standard.csv"
    other_path = other_dir / "full_deduplicated_standard.csv"
    output_path = PROCESSED_PATH / "descriptors" / "merged.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.merge(
        pd.read_csv(main_path),
        pd.read_csv(other_path),
        how="outer",
        on="SMILES"
    ).to_csv(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()