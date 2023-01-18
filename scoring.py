"""Scoring function to calculate custom scores of Fairseq models
SMILES or SELFIES, 2022
"""

import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.bart import BARTModel
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
from tqdm import tqdm

from constants import MOLNET_DIRECTORY, TASK_MODEL_PATH, TASK_PATH

CUDA_DEVICE = 3


def load_model(model_path: Path, data_path: Path, cuda_device: str = None):
    """Load fairseq BART model

    Args:
        model_path (Path): path to .pt file
        cuda_device (str, optional): if model should be converted to a device. Defaults to None.

    Returns:
        fairseq_model: load BART model
    """
    model = BARTModel.from_pretrained(
        str(model_path.parent),
        data_name_or_path=str(data_path),
        checkpoint_file=str(model_path.name),
    )
    model.eval()
    if cuda_device:
        model.cuda(device=cuda_device)
    return model


def load_dataset(data_path: Path, classification: bool = True) -> List[str]:
    """Load dataset with fairseq

    Args:
        data_path (Path): folder path of data (e.g. /input0/test)
        classification (bool): if classification(True) or regression(False) loading should be used. Defaults to classification.


    Returns:
        List[str]: loaded fairseq dataset
    """
    if classification:
        dikt = Dictionary.load(str(data_path.parent / "dict.txt"))
        data = list(load_indexed_dataset(str(data_path), dikt))
        return data
    with open(data_path, "r") as label_file:
        label_lines = label_file.readlines()
    return [float(line.strip()) for line in label_lines]


def get_predictions(
    model,
    mols: np.ndarray,
    targets: np.ndarray,
    target_dict_path: Path,  # maybe None for classifications?
    classification: bool = True,
) -> Tuple[List[float], List[float]]:
    """Get predictions of model on mols

    Args:
        model (fairseq_model): fairseq model to make predictions with
        mols (np.ndarray): dataset to make predictions
        targets (np.ndarray): targets to predict against
        target_dict_path (Path): path to target_dict to translate model output to class
        classification (bool): if classification(True) or regression(False). Defaults to classification.

    Returns:
        Tuple[List[float], List[float]]: predictions, targets
    """
    # from https://github.com/YerevaNN/BARTSmiles/blob/main/evaluation/compute_score.py
    preds = []
    seen_targets = []
    if classification:
        target_dict = Dictionary.load(str(target_dict_path))
    for (smile, target) in tqdm(list(zip(mols, targets))):
        smile = torch.cat(
            (torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2]))
        )
        output = model.predict(
            "sentence_classification_head", smile, return_logits=not classification
        )
        if classification:
            target = target[0].item()
            if target_dict[4] == "1":
                preds.append(output[0][0].exp().item())
                seen_targets.append(-1 * target + 5)
            else:
                preds.append(output[0][1].exp().item())
                seen_targets.append(target - 4)
        else:
            preds.append(output[0][0].item())
            seen_targets.append(target)
    return preds, seen_targets


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
    subpaths = [Path(subpath_string) for subpath_string in subpath_strings]
    names = [subpath.name for subpath in subpaths]
    return zip(subpaths, names)


def parse_hyperparams(param_string: str) -> Dict[str, str]:
    """Parse hyperparameter string

    Args:
        param_string (str): parameter string to parse

    Returns:
        Dict[str, str]: dictionary with hyperparameters
    """
    param_parts = param_string.split("_")
    output = {
        "learning_rate": param_parts[0],
        "dropout": param_parts[1],
        "model_size": param_parts[2],
        "data_type": param_parts[3],
    }
    return output


if __name__ == "__main__":
    for task_path, task in iterate_paths(TASK_MODEL_PATH):
        if task not in MOLNET_DIRECTORY:
            continue
        classification = MOLNET_DIRECTORY[task]["dataset_type"] == "classification"
        for tokenizer_path, tokenizer in iterate_paths(task_path):
            for hyperparameter_path, hyperparameter in iterate_paths(tokenizer_path):
                if (hyperparameter_path / hyperparameter).exists():
                    best_checkpoint_path = (
                        hyperparameter_path / hyperparameter / "checkpoint_best.pt"
                    )
                else:
                    best_checkpoint_path = hyperparameter_path / "checkpoint_best.pt"
                if not best_checkpoint_path.is_file():
                    continue
                model = load_model(
                    best_checkpoint_path, TASK_PATH / task / tokenizer, CUDA_DEVICE
                )
                mols = load_dataset(TASK_PATH / task / tokenizer / "input0" / "test")
                if classification:
                    labels = load_dataset(
                        TASK_PATH / task / tokenizer / "label" / "test", classification
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
                score_dict, report = get_score(preds, seen_targets, classification)
                score_dict = score_dict | parse_hyperparams(hyperparameter)
                if classification:
                    with open(hyperparameter_path / "report.txt", "w") as report_file:
                        report_file.write(report)
                else:
                    with open(hyperparameter_path / "report.txt", "w") as report_file:
                        report_file.write(json.dumps(report, indent=4))
                score_dict["task_type"] = (
                    "classification" if classification else "regression"
                )
                score_dict["task"] = task
                score_dict["tokenizer"] = tokenizer
                pd.DataFrame([score_dict]).to_csv(hyperparameter_path / "scores.csv")
