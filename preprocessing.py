""" Preprocessing 
SMILES or SELFIES, 2022
"""
import logging
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import selfies
from rdkit import Chem

from constants import CALCULATOR, DESCRIPTORS, PROJECT_PATH


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


def check_valid(input: str) -> bool:
    """Check validity of SMILES string
    https://github.com/rdkit/rdkit/issues/2430

    Args:
        input (str): SMILES string

    Returns:
        bool: True if valid, False if invalid
    """
    m = Chem.MolFromSmiles(input)
    # check if syntactically correct and rdkit valid
    return False if m is None else True


def canonize_smile(input: str) -> str:
    """Canonize SMILES string

    Args:
        input (str): SMILES input string

    Returns:
        str: canonize SMILES string
    """
    return Chem.CanonSmiles(input)


def check_canonized(input: str) -> bool:
    """Check if a (SMILES-)string is already canonized

    Args:
        input (str): (SMILES-)string to check

    Returns:
        bool: True if is canonized
    """
    return canonize_smile(input) == input


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
    return np.array(list(descriptors.values())), "valid"


def process_mol_file(input_file: Path) -> Tuple[List[dict], dict]:
    statistics = {}
    output = []
    with open(input_file, "r") as open_file:
        smiles = open_file.readlines()
    for smile in smiles:
        processed, validity = process_mol(smile)
        if processed is not None:
            output.append(processed)
        statistics[validity] = statistics.get(validity, 0) + 1
    return (output, statistics)


def process_mol_files(
    input_folder: Path, max_processes: int = 4
) -> Tuple[pd.DataFrame, dict]:
    """Process all files in input folder

    Args:
        input_folder (Path): (parent-)folder, where every file should be inspected

    Returns:
        Tuple[pd.DataFrame, dict]:  descriptors of all valid molecules in the files
                                    dict of absolute number of filtering statistics
    """
    output = []
    statistics = {}
    input_files = list(input_folder.glob("*.txt"))
    with Pool(min(max_processes, len(input_files))) as pool:
        for result in pool.map(
            func=process_mol_file, iterable=input_files, chunksize=None
        ):
            curr_output, curr_statistics = result
            output.extend(curr_output)
            for key in curr_statistics:
                statistics[key] = statistics.get(key, 0) + curr_statistics[key]

    return (
        pd.DataFrame.from_records(
            output, columns=DESCRIPTORS + ["SELFIE", "SELFIE_LENGTH", "SMILE"]
        ),
        statistics,
    )


if __name__ == "__main__":
    processed_mols, statistics = process_mol_files(PROJECT_PATH / "download_10m")
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
    (PROJECT_PATH / "processed").mkdir(exist_ok=True)
    processed_mols.to_csv(PROJECT_PATH / "processed" / "10m_pubchem.csv")
