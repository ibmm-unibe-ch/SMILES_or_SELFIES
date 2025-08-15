"""Generic utils functions
SMILES or SELFIES, 2023
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

def pickle_object(obj: Any, path: Path) -> None:
    """Serialize and save an object to disk using pickle.

    Args:
        obj: Python object to be serialized. Must be pickleable.
        path: File path where the object will be saved. Parent directories
              will be created if they don't exist.

    Raises:
        pickle.PicklingError: If the object cannot be pickled.
        OSError: If the file cannot be written.

    Example:
        >>> data = {"key": "value"}
        >>> pickle_object(data, Path("data.pkl"))
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def unpickle(path: Path) -> Any:
    """Load and deserialize a pickled object from disk.

    Args:
        path: File path containing the pickled object.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the specified path doesn't exist.
        pickle.UnpicklingError: If the file cannot be unpickled.

    Example:
        >>> data = unpickle(Path("data.pkl"))
    """
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def log_and_add(text: str, message: str) -> str:
    """Log a message and append it to an existing text string.

    Args:
        text: The existing text string to append to.
        message: The message to log and append.

    Returns:
        The input text with the message appended.

    Example:
        >>> log = "Start processing\\n"
        >>> log = log_and_add(log, "Processing complete")
        Start processing
        Processing complete
    """
    logging.info(message)
    return f"{text}{message}\n"


def parse_arguments(
    cuda: bool = False,
    tokenizer: bool = False,
    task: bool = False,
    seeds: bool = False,
    dict_params: bool = False,
    model_type: bool = False
) -> Dict[str, Optional[str]]:
    """Parse command line arguments with flexible configuration.

    Args:
        cuda: Whether to include CUDA device argument. Defaults to False.
        tokenizer: Whether to include tokenizer argument. Defaults to False.
        task: Whether to include task argument. Defaults to False.
        seeds: Whether to include seeds argument. Defaults to False.
        dict_params: Whether to include hyperparameters dict argument. Defaults to False.
        model_type: Whether to include model type argument. Defaults to False.

    Returns:
        Dictionary of parsed arguments where keys are argument names and values
        are the provided values (or None if not provided).

    Example:
        >>> args = parse_arguments(cuda=True, task=True)
        # Can then access args['cuda'] and args['task']
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for SMILES/SELFIES processing"
    )

    if cuda:
        parser.add_argument(
            "--cuda",
            required=True,
            help="VISIBLE_CUDA_DEVICE (e.g., '0' or '1')"
        )
    if task:
        parser.add_argument(
            "--task",
            help="Specific task to process (e.g., 'bbbp' or 'tox21')"
        )
    if tokenizer:
        parser.add_argument(
            "--tokenizer",
            help="Tokenizer configuration to use (e.g., 'smiles_atom_isomers')"
        )
    if seeds:
        parser.add_argument(
            "--seeds",
            help="Number of different random seeds to use for experiments"
        )
    if dict_params:
        parser.add_argument(
            "--dict",
            help="Dictionary of hyperparameters from hyperparams.py"
        )
    if model_type:
        parser.add_argument(
            "--modeltype",
            help="Model architecture type (e.g., 'roberta' or 'bart')"
        )

    return vars(parser.parse_args())