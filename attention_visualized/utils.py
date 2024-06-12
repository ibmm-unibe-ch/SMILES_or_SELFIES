"""Generic utils functions
SMILES or SELFIES, 2023
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

def pickle_object(objekt, path: Path):
    """Pickle an object at *path*

    Args:
        objekt: object to pickle
        path (Path): path to save to
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "wb") as openfile:
        pickle.dump(objekt, openfile)


def unpickle(path: Path):
    """Unpickle an object from path.

    Args:
        path (Path): Path to unpickle from

    Returns:
        object: pickled object
    """
    path = Path(path)
    with open(path, "rb") as openfile:
        objekt = pickle.load(openfile)
    return objekt


def log_and_add(text: str, string: str) -> str:
    """Log string and add it to text

    Args:
        text (str): Longer text to add string to
        string (str): String to add to log and add to text

    Returns:
        str: Extended text
    """
    logging.info(string)
    text += string + "\n"
    return text


def parse_arguments(cuda=False, tokenizer=False, task=False, seeds=False, dict_params=False, model_type=False) -> dict:
    """Parse command line arguments

    Args:
        cuda (bool, optional): Should CUDA be selectable? Defaults to False.
        tokenizer (bool, optional): Should tokenizer be selectable? Defaults to False.
        task (bool, optional): Should task be selectable? Defaults to False.

    Returns:
        dict: Dict with all selected arguments.
    """
    parser = argparse.ArgumentParser()
    if cuda:
        parser.add_argument("--cuda", required=True, help="VISIBLE_CUDA_DEVICE")
    if task:
        parser.add_argument("--task", help="Which specific task as string.")
    if tokenizer:
        parser.add_argument("--tokenizer", help="Which tokenizer to use.")
    if seeds:
        parser.add_argument("--seeds", help="How many different random seeds should be used.")
    if dict_params:
        parser.add_argument("--dict", help="Optional dict of hyperparameters stored in hyperparams.py.")
    if model_type:
        parser.add_argument("--modeltype", help="Optional dict of hyperparameters stored in hyperparams.py.") 
    args = parser.parse_args()
    return vars(args)