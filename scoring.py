"""Scoring function to calculate custom scores of Fairseq models
SMILES or SELFIES, 2022
"""

import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
from constants import MOLNET_DIRECTORY, TASK_MODEL_PATH, TASK_PATH
from fairseq_utils import get_predictions, load_dataset, load_model
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    root_mean_squared_error,
)
from utils import parse_arguments


def get_score(
    predictions: List[float],
    seen_targets: List[float],
    classification: bool = True
) -> Tuple[Dict[str, float], Union[str, Dict[str, float]]]:
    """Compute evaluation metrics for model predictions.

    Args:
        predictions: Model predictions as probabilities or continuous values.
        seen_targets: Ground truth labels or values.
        classification: Whether to compute classification metrics (True) or
                       regression metrics (False). Defaults to True.

    Returns:
        A tuple containing:
        - Dictionary of metric names and values
        - Classification report (str) for classification tasks or
          metric dictionary for regression tasks
    """
    score_dict: Dict[str, float] = {}

    if classification:
        predicted_classes = [int(prediction >= 0.5) for prediction in predictions]
        score_dict["ROC_AUC"] = roc_auc_score(seen_targets, predictions)
        score_dict["average_precision"] = average_precision_score(seen_targets, predictions)
        score_dict["F1_score"] = f1_score(seen_targets, predicted_classes)
        score_dict["accuracy_score"] = accuracy_score(seen_targets, predicted_classes)
        report = classification_report(seen_targets, predicted_classes)
        return score_dict, report
    else:
        score_dict["mean_absolute_error"] = mean_absolute_error(seen_targets, predictions)
        score_dict["max_error"] = max_error(seen_targets, predictions)
        score_dict["mean_squared_error"] = mean_squared_error(seen_targets, predictions)
        score_dict["rectified_mean_squared_error"] = root_mean_squared_error(seen_targets, predictions)
        return score_dict, score_dict


def iterate_paths(path: Path) -> List[Tuple[Path, str]]:
    """Find all subdirectories in a given path and return their names.

    Args:
        path: Directory path to search for subdirectories.

    Returns:
        List of tuples containing:
        - Path to subdirectory
        - Name of the subdirectory (last component of path)
    """
    subpath_strings = glob(str(path) + "/*", recursive=True)
    subpaths = list(reversed([Path(subpath_string) for subpath_string in subpath_strings]))
    names = [subpath.name for subpath in subpaths]
    return list(zip(subpaths, names))


def parse_hyperparams(param_string: str, use_seed: bool) -> Dict[str, Union[float, str]]:
    """Parse hyperparameter string into a dictionary.

    Args:
        param_string: String containing hyperparameters in format "lr_dropout_seed"
        use_seed: Whether to include seed in the output dictionary.

    Returns:
        Dictionary containing parsed hyperparameters with keys:
        - learning_rate (float)
        - dropout (float)
        - seed (str, optional if use_seed is True and present in param_string)
    """
    param_parts = param_string.split("_")
    output = {
        "learning_rate": float(param_parts[0]),
        "dropout": float(param_parts[1])
    }
    if use_seed and len(param_parts) >= 4:
        output["seed"] = param_parts[3]
    return output


def parse_tokenizer(tokenizer_string: str) -> Dict[str, str]:
    """Parse tokenizer configuration string into components.

    Args:
        tokenizer_string: String containing tokenizer configuration in format
                         "embedding_tokenizer_dataset_architecture"

    Returns:
        Dictionary containing parsed tokenizer configuration with keys:
        - embedding (str): Type of molecular representation (e.g., "smiles")
        - tokenizer (str): Tokenizer type (e.g., "atom")
        - dataset (str): Dataset variant (e.g., "isomers")
        - architecture (str): Model architecture (defaults to "bart")
    """
    tokenizer_parts = tokenizer_string.split("_")
    output = {
        "embedding": tokenizer_parts[0],
        "tokenizer": tokenizer_parts[1],
        "dataset": tokenizer_parts[2],
        "architecture": tokenizer_parts[3] if len(tokenizer_parts) > 3 else "bart",
    }
    return output


def evaluate_model(
    model_path: Path,
    task_path: Path,
    tokenizer: str,
    cuda: str,
    classification: bool
) -> Tuple[List[float], List[float]]:
    """Load and evaluate a model on test data.

    Args:
        model_path: Path to the model checkpoint.
        task_path: Path to the task directory containing test data.
        tokenizer: Tokenizer configuration string.
        cuda: CUDA device identifier.
        classification: Whether the task is classification.

    Returns:
        Tuple containing:
        - Model predictions
        - Ground truth labels/values
    """
    model = load_model(model_path, task_path / tokenizer, cuda)
    mols = load_dataset(task_path / tokenizer / "input0" / "test")
    
    if classification:
        labels = load_dataset(
            task_path / tokenizer / "label" / "test",
            classification
        )
    else:
        labels = load_dataset(
            task_path / tokenizer / "label" / "test.label",
            classification
        )
    
    return get_predictions(
        model,
        mols,
        labels,
        task_path / tokenizer / "label" / "dict.txt",
        classification
    )


def process_hyperparameter_directory(
    hyperparameter_path: Path,
    hyperparameter: str,
    task: str,
    config: str,
    cuda: str,
    use_seed: bool
) -> Optional[Dict[str, Union[str, float]]]:
    """Process a single hyperparameter directory and evaluate the model.

    Args:
        hyperparameter_path: Path to the hyperparameter directory.
        hyperparameter: Hyperparameter configuration string.
        task: Name of the task being evaluated.
        config: Model configuration string.
        cuda: CUDA device identifier.
        use_seed: Whether to include seed in the output.

    Returns:
        Dictionary containing evaluation results, or None if model couldn't be evaluated.
    """
    # Determine the path to the best checkpoint
    if (hyperparameter_path / hyperparameter).exists():
        best_checkpoint_path = hyperparameter_path / hyperparameter / "checkpoint_best.pt"
    else:
        best_checkpoint_path = hyperparameter_path / "checkpoint_best.pt"

    if not best_checkpoint_path.is_file(): # or (use_seed and not("seed" in str(hyperparameter_path))) or (use_seed and not Path(str(best_checkpoint_path.parent)[:-1]+seeds).exists()) or (not use_seed and ("seed" in str(hyperparameter_path))):
        print(f"Skipping hyperparameter {hyperparameter_path}")
        return None

    print(f"Working hyperparameter {hyperparameter_path}")
    classification = MOLNET_DIRECTORY[task]["dataset_type"] == "classification"
    tokenizer = "_".join(config.split("_")[:3])

    try:
        preds, seen_targets = evaluate_model(
            best_checkpoint_path,
            TASK_PATH / task,
            tokenizer,
            cuda,
            classification
        )
    except Exception as e:
        print(f"Error evaluating {hyperparameter_path}: {str(e)}")
        return None

    output_dict = {
        "task": task,
        "task_type": "classification" if classification else "regression",
    }
    score_dict, report = get_score(preds, seen_targets, classification)
    
    # Combine all information
    result_dict = {
        **output_dict,
        **parse_tokenizer(config),
        **parse_hyperparams(hyperparameter, use_seed),
        **score_dict
    }

    # Save reports
    report_path = hyperparameter_path / "report.txt"
    with open(report_path, "w") as report_file:
        report_file.write(report if classification else json.dumps(report, indent=4))

    # Save scores
    scores_path = hyperparameter_path / "scores.csv"
    pd.DataFrame([result_dict]).to_csv(scores_path)

    return result_dict


def main() -> None:
    """Main function to evaluate all models in the task directory."""
    arguments = parse_arguments(cuda=True, seeds=True)
    cuda = arguments["cuda"]
    seeds = arguments["seeds"]
    use_seed = int(seeds) > 1

    all_results = []

    for task_path, task in iterate_paths(TASK_MODEL_PATH):
        for config_path, config in iterate_paths(task_path):
            for hyperparameter_path, hyperparameter in iterate_paths(config_path):
                result = process_hyperparameter_directory(
                    hyperparameter_path,
                    hyperparameter,
                    task,
                    config,
                    cuda,
                    use_seed
                )
                if result:
                    all_results.append(result)

if __name__ == "__main__":
    main()