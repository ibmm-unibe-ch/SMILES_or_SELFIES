"""Generic utils functions
SMILES or SELFIES, 2023
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from ast import literal_eval
import re
from statistics import mean, stdev

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

def clean_string(string: str) -> str:
    """Clean and normalize a string containing parameter data.
    
    Args:
        string: Input string to clean.
        
    Returns:
        Cleaned string with array notation and parameter artifacts removed.
    """
    # Remove parameter artifacts
    string = re.sub(r"\'param\_.*\)\,\'params\'", "'params'", string)
    # Convert numpy array notation to standard list notation
    string = string.replace("array([", "[").replace("])", "]")
    # Remove dtype specifications
    string = re.sub(r"\],\s*dtype=.*\),", "],", string)
    return string

def reading_dict(path: Union[str, Path]) -> tuple[str, dict]:
    """Read and parse a dictionary file containing estimator parameters.
    
    Args:
        path: Path to the dictionary file.
        
    Returns:
        Tuple containing estimator name and parsed parameters dictionary.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file content cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # First line contains estimator name, rest contains parameters
        estimator_name = lines[0].strip()
        param_string = "".join(line.strip() for line in lines[1:])
        cleaned_string = clean_string(param_string)
        params_dict = literal_eval(cleaned_string)
        
        return estimator_name, params_dict
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Failed to parse dictionary file {path}: {e}")


def get_cells(line: str) -> List[str]:
    """Split a line into cells separated by multiple spaces.
    
    Args:
        line: Input line to split.
        
    Returns:
        List of non-empty cells.
    """
    return [cell.strip() for cell in re.split(r"\s{2,}", line) if cell.strip()]


def parse_dict(cv_results: Dict[str, Any]) -> tuple[Dict[str, Any], float, float]:
    """Parse cross-validation results to extract best parameters and scores.
    
    Args:
        cv_results: Dictionary containing cross-validation results.
        
    Returns:
        Tuple containing best parameters, mean score, and standard deviation.
        
    Raises:
        ValueError: If the results dictionary is malformed.
    """
    try:
        best_model_index = cv_results['rank_test_score'].index(1)
        best_params = cv_results["params"][best_model_index]
        
        best_scores = []
        for i in range(3):  # Assuming 3-fold cross-validation
            score_key = f"split{i}_test_score"
            if score_key in cv_results:
                best_scores.append(abs(float(cv_results[score_key][best_model_index])))
        
        if not best_scores:
            raise ValueError("No test scores found in CV results")
        
        return best_params, mean(best_scores), stdev(best_scores)
    
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Malformed CV results dictionary: {e}")


def get_report(path: Union[str, Path]) -> Dict[str, Any]:
    """Generate a report from a cross-validation results file.
    
    Args:
        path: Path to the results file.
        
    Returns:
        Dictionary containing estimator name, best parameters, and scores.
    """
    estimator, cv_results = reading_dict(path)
    best_params, mean_score, std_score = parse_dict(cv_results)
    
    return {
        "estimator": estimator,
        "best_params": best_params,
        "best_scores": mean_score,
        "std": std_score,
    }

def parse_tokenizer(tokenizer_string: str) -> Dict[str, str]:
    """Parse tokenizer configuration string into components.
    
    Args:
        tokenizer_string: Tokenizer string to parse (e.g., 'smiles_atom_isomers').
        
    Returns:
        Dictionary with tokenizer settings.
        
    Raises:
        ValueError: If the tokenizer string format is invalid.
    """
    tokenizer_parts = tokenizer_string.split("_")
    if len(tokenizer_parts) < 3:
        raise ValueError(f"Invalid tokenizer string format: {tokenizer_string}")
    
    return {
        "embedding": tokenizer_parts[0],
        "tokenizer": tokenizer_parts[1],
        "dataset": tokenizer_parts[2],
    }

def compute_zscore(
    column: Any, 
    value: Optional[float] = None
) -> Union[float, Any]:
    """Compute z-score for a value or an entire column.
    
    Args:
        column: Data column (pandas Series, list, or array-like).
        value: Specific value to compute z-score for. If None, computes for entire column.
        
    Returns:
        Z-score(s) for the input data.
    """
    if value:
        return (value-column.mean())/column.std()
    return (column-column.mean())/column.std()