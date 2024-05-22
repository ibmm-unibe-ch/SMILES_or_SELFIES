"""Scoring function to calculate custom scores of Fairseq models
SMILES or SELFIES, 2022
"""

import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from constants import (
    MOLNET_DIRECTORY,
    TASK_MODEL_PATH,
    TASK_PATH,
)
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
)
from utils import parse_arguments


def get_score(
    predictions: List[float], seen_targets: List[float], classification: bool = True
) -> Tuple[dict, str]:
    """Compute scores of predictions and seen_targets

    Args:
        predictions (List[float]): predictions made by model
        seen_targets (List[float]): ground_truth models
        classification (bool, optional): whether classification metrics should be used (True) or regression metrics (False). Defaults to True.

    Returns:
        Tuple[dict, str]: score_dictionary, report
    """
    score_dikt = {}
    if classification:
        predicted_classes = [int(prediction >= 0.5) for prediction in predictions]
        roc_auc = roc_auc_score(seen_targets, predictions)
        score_dikt["ROC_AUC"] = roc_auc
        average_precision = average_precision_score(seen_targets, predictions)
        score_dikt["average_precision"] = average_precision
        f1 = f1_score(seen_targets, predicted_classes)
        score_dikt["F1_score"] = f1
        acc_score = accuracy_score(seen_targets, predicted_classes)
        score_dikt["accuracy_score"] = acc_score
        report = classification_report(seen_targets, predicted_classes)
        return score_dikt, report
    mae = mean_absolute_error(seen_targets, predictions)
    score_dikt["mean_absolute_error"] = mae
    max_err = max_error(seen_targets, predictions)
    score_dikt["max_error"] = max_err
    mse = mean_squared_error(seen_targets, predictions)
    score_dikt["mean_squared_error"] = mse
    rmse = mean_squared_error(seen_targets, predictions, squared=False)
    score_dikt["rectified_mean_squared_error"] = rmse
    return score_dikt, score_dikt


def iterate_paths(path: Path) -> List[Tuple[Path, str]]:
    """Iterate paths with glob and return names

    Args:
        path (Path): Path to search for subpaths

    Returns:
        List[Tuple[Path, str]]: subpaths, name of last bit of subpaths
    """
    subpath_strings = glob(str(path) + "/*", recursive=True)
    subpaths = list(reversed([Path(subpath_string) for subpath_string in subpath_strings]))
    names = [subpath.name for subpath in subpaths]
    out = zip(subpaths, names)
    return out


def parse_hyperparams(param_string: str, use_seed:bool) -> Dict[str, str]:
    """Parse hyperparameter string

    Args:
        param_string (str): parameter string to parse

    Returns:
        Dict[str, str]: dictionary with hyperparameters
    """
    param_parts = param_string.split("_")
    output = {
        "learning_rate": float(param_parts[0]),
        "dropout": float(param_parts[1])
    }
    if use_seed and len(param_parts)>=4:
        output["seed"] = param_parts[3]
    return output

def parse_tokenizer(tokenizer_string:str) -> Dict[str, str]:
    """Parse tokenizer string

    Args:
        tokenizer_string (str): tokenizer string to parse

    Returns:
        Dict[str, str]: dictionary with tokenizer settings
    """
    tokenizer_parts = tokenizer_string.split("_")
    output = {
        "embedding": tokenizer_parts[0],
        "tokenizer": tokenizer_parts[1],
        "dataset": tokenizer_parts[2],
        "architecture": tokenizer_parts[3] if len(tokenizer_parts)>3 else "bart",
    }
    return output

if __name__ == "__main__":
    arguments = parse_arguments(True, False, False, True, False, False)
    cuda = arguments["cuda"]
    seeds = arguments["seeds"]
    use_seed = int(seeds) > 1
    for task_path, task in iterate_paths(TASK_MODEL_PATH):
        for config_path, config in iterate_paths(task_path):          
            if task in MOLNET_DIRECTORY:
                classification = MOLNET_DIRECTORY[task]["dataset_type"] == "classification"                
                for hyperparameter_path, hyperparameter in iterate_paths(config_path):
                    if (hyperparameter_path / hyperparameter).exists():
                        best_checkpoint_path = hyperparameter_path / hyperparameter / "checkpoint_best.pt"
                    else:
                        best_checkpoint_path = (hyperparameter_path / "checkpoint_best.pt")
                    if not best_checkpoint_path.is_file():# or (use_seed and not("seed" in str(hyperparameter_path))) or (use_seed and not Path(str(best_checkpoint_path.parent)[:-1]+seeds).exists()) or (not use_seed and ("seed" in str(hyperparameter_path))):
                        print(f"Skipping hyperparameter {hyperparameter_path}")
                        continue
                    print(f"Working hyperparameter {hyperparameter_path}")
                    tokenizer = "_".join(config.split("_")[:3])
                    model = load_model(
                        best_checkpoint_path, TASK_PATH / task / tokenizer, cuda
                    )
                    mols = load_dataset(
                        TASK_PATH / task / tokenizer / "input0" / "test"
                    )
                    if classification:
                        labels = load_dataset(
                            TASK_PATH / task / tokenizer / "label" / "test",
                            classification,
                        )
                    else:
                        labels = load_dataset(
                            TASK_PATH / task / tokenizer / "label" / "test.label",
                            classification,
                        )
                    preds, seen_targets = get_predictions(
                        model,
                        mols,
                        labels,
                        TASK_PATH / task / tokenizer / "label" / "dict.txt",
                        classification,
                    )
                    output_dict = {
                        "task": task,
                        "task_type": "classification"
                        if classification
                        else "regression",
                    }
                    score_dict, report = get_score(preds, seen_targets, classification)
                    output_dict = (
                        output_dict | parse_tokenizer(config) | parse_hyperparams(hyperparameter, use_seed=use_seed) | score_dict
                    )
                    if classification:
                        with open(
                            hyperparameter_path / "report.txt", "w"
                        ) as report_file:
                            report_file.write(report)
                    else:
                        with open(
                            hyperparameter_path / "report.txt", "w"
                        ) as report_file:
                            report_file.write(json.dumps(report, indent=4))
                    pd.DataFrame([output_dict]).to_csv(
                        hyperparameter_path / "scores.csv"
                    )
