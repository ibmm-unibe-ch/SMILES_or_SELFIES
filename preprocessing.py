""" Preprocessing
SMILES or SELFIES, 2022
"""
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import selfies
from rdkit import Chem
from tqdm import tqdm

from constants import CALCULATOR, DESCRIPTORS, PROCESSED_PATH, PROJECT_PATH


def calc_descriptors(mol_string: str) -> dict:
    """Calculate the descriptors (features) of the given molecule

    Args:
        mol_string (str): molecule given as SMILES string

    Returns:
        dict: {descriptor_name : value}
    """
    mol = Chem.MolFromSmiles(mol_string)
    if mol is None:
        return {key: None for key in DESCRIPTORS}
    calcs = CALCULATOR.CalcDescriptors(mol)
    return {key: value for key, value in zip(DESCRIPTORS, calcs)}


def check_valid(input_str: str) -> bool:
    """Check validity of SMILES string
    https://github.com/rdkit/rdkit/issues/2430

    Args:
        input_str (str): SMILES string

    Returns:
        bool: True if valid, False if invalid
    """
    m = Chem.MolFromSmiles(input_str)
    # check if syntactically correct and rdkit valid
    return m is not None


def canonize_smile(input_str: str) -> str:
    """Canonize SMILES string

    Args:
        input_str (str): SMILES input string

    Returns:
        str: canonize SMILES string
    """
    return Chem.CanonSmiles(input_str)


def check_canonized(input_str: str) -> bool:
    """Check if a (SMILES-)string is already canonized

    Args:
        input_str (str): (SMILES-)string to check

    Returns:
        bool: True if is canonized
    """
    return canonize_smile(input_str) == input_str


def translate_selfie(smile: str) -> Tuple[str, int]:
    """Translate SMILES to SELFIES

    Args:
        smile (str): input string as SMILES

    Returns:
        Tuple[str, int]: SELFIES string, length of SELFIES string
    """
    try:
        selfie = selfies.encoder(smile)
        len_selfie = selfies.len_selfies(selfie)
        return (selfie, len_selfie)
    except Exception as e:
        logging.error(e)
        return (None, -1)


def process_mol(mol: str) -> Tuple[dict, str]:
    """Process a single molecule given as SMILES string

    Args:
        mol (str): SMILES string of molecule to inspect

    Returns:
        Tuple[dict, str]:   dict with properties including SELFIES and canonized SMILES
                            class of error
    """
    if not check_valid(mol):
        logging.warning(f"The following sequence was deemed invalid by RdKit: {mol}")
        return None, "invalid_smile"
    canon = canonize_smile(mol)
    descriptors = calc_descriptors(canon)
    selfie, selfie_length = translate_selfie(canon)
    if selfie_length == -1:
        return None, "invalid_selfie"
    descriptors["SELFIE"] = selfie
    descriptors["SELFIE_LENGTH"] = selfie_length
    descriptors["SMILE"] = canon
    return descriptors.values(), "valid"


def process_mol_file(input_file: Path) -> Tuple[List[dict], dict]:
    """Processing of a single file with many molecules

    Args:
        input_file (Path): input file to process

    Returns:
        Tuple[List[dict], dict]: processed mol file, statistics-dict
    """
    statistics = {}
    output = []
    with open(input_file, "r") as open_file:
        smiles = open_file.readlines()
    for smile in tqdm(smiles):
        processed, validity = process_mol(smile)
        if processed is not None:
            output.append(processed)
        statistics[validity] = statistics.get(validity, 0) + 1
    return (output, statistics)


def process_mol_files(input_folder: Path) -> Tuple[pd.DataFrame, dict]:
    """Process all files in input folder

    Args:
        input_folder (Path): (parent-)folder, where every file should be inspected

    Returns:
        Tuple[pd.DataFrame, dict]:  descriptors of all valid molecules in the files
                                    dict of absolute number of filtering statistics
    """
    paths = []
    statistics = {}
    input_files = list(input_folder.glob("*"))
    for input_file in input_files:
        result = process_mol_file(input_file)
        curr_output, curr_statistics = result
        curr_path = PROCESSED_PATH / "".join(
            [letter for letter in str(uuid.uuid4()) if letter.isalnum()]
        )
        curr_df = pd.DataFrame(curr_output)
        curr_df.to_csv(curr_path)
        paths.append(curr_path)
        for key in curr_statistics:
            statistics[key] = statistics.get(key, 0) + curr_statistics[key]
    return (paths, statistics)


def merge_dataframes(
    paths_path: Path, output_path: Path, remove_paths: bool = False
) -> pd.DataFrame:
    """Merge the dataframes saved at paths_path and save to output_path

    Args:
        paths_path (Path): paths were base dataframes are
        output_path (Path): path to save merged dataframe
        remove_paths (bool, optional): flag to remove base dataframes. Defaults to False.

    Returns:
        pd.DataFrame: merged DataFrame
    """
    dataframes = []
    with open(paths_path, "rb") as handle:
        paths = pickle.load(handle)
    logging.info(f"Merging together {len(paths)} dataframes.")
    for path in paths:
        curr_dataframe = pd.read_csv(path)
        dataframes.append(curr_dataframe)
    merged_dataframe = pd.concat(dataframes)
    merged_dataframe.to_csv(output_path)
    if remove_paths:
        for path in paths:
            os.remove(path)
    return merged_dataframe


def check_dups(df: pd.DataFrame) -> pd.DataFrame:
    """Check the duplicates of DataFrame

    Args:
        df (pd.DataFrame): DataFrame to check

    Returns:
        pd.DataFrame: de-duplicated DataFrame
    """
    logging.info("Checking for SMILES duplicates.. ")
    print(df.head())
    duplicated_rows = df.duplicated(subset=["210"])

    dup_numbers = df.duplicated(subset=["210"]).sum()
    logging.info(
        "{} SMILES duplicates found.. ".format(dup_numbers)
    )  # duplicates are correctly detected, has been tested
    return df[~duplicated_rows]  # duplicates correctly removed from df, tested


if __name__ == "__main__":
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    paths, statistics = process_mol_files(PROJECT_PATH / "download_10m")
    invalid_smile = statistics.get("invalid_smile", 0)
    invalid_selfie = statistics.get("invalid_selfie", 0)
    valid = statistics.get("valid", 0)
    all_mols = invalid_smile + invalid_selfie + valid
    curr_mols = all_mols
    logging.info(
        f"There were {invalid_smile} invalid SMILES found and {curr_mols-invalid_smile} passed this stage."
    )
    logging.info(
        f"This amounts to a percentage of {100*invalid_smile/(curr_mols):.2f}."
    )
    curr_mols = curr_mols - invalid_smile
    logging.info(
        f"There were {invalid_selfie} invalid SELFIES found and {curr_mols-invalid_selfie} passed this stage."
    )
    logging.info(
        f"This amounts to a percentage of {100*invalid_selfie/(curr_mols):.2f}."
    )
    curr_mols = curr_mols - invalid_selfie
    logging.info(f"We filtered out {all_mols-curr_mols} many samples.")
    logging.info(f"This amounts to a percentage of {100*(1-curr_mols/all_mols):.2f}.")
    logging.info(f"The arrays are saved in {paths}")
    with open(PROCESSED_PATH / "paths.pickle", "wb") as handle:
        pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
    PROCESSED_PATH.mkdir(exist_ok=True)
    merged_dataframe = merge_dataframes(
        PROCESSED_PATH / "paths.pickle", PROCESSED_PATH / "10m_pubchem.csv", True
    )
    deduplicated_dataframe = check_dups(merged_dataframe)
    deduplicated_dataframe.to_csv(PROCESSED_PATH / "10m_deduplicated.csv")
