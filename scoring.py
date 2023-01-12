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

from constants import MOLNET_DIRECTORY, TASK_PATH


def load_model(model_path: Path, cuda_device: str = None):
    model = BARTModel.from_pretrained(
        str(model_path.parent), checkpoint_file=str(model_path.name)
    )
    model.eval()
    if cuda_device:
        model.cuda(device=cuda_device)
    return model


def load_dataset(data_path: Path):
    dikt = Dictionary.load(str(data_path.parent / "dict.txt"))
    data = list(load_indexed_dataset(str(data_path), dikt))
    return data


def load_regression_dataset(data_path: Path):
    with open(data_path, "r") as label_file:
        label_lines = label_file.readlines()
    return [float(line.strip()) for line in label_lines]


def get_predictions(
    model,
    mols: np.ndarray,
    targets: np.ndarray,
    target_dict_path: Dict,  # maybe None for classifications?
    classification: bool = True,
) -> Tuple[List[float], List[float]]:
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


if __name__ == "__main__":
    cuda_device = 3
    model_type = "base"
    for tokenizer_suffix_path in glob(str(TASK_PATH) + "/*", recursive=True):
        tokenizer_suffix_path = Path(tokenizer_suffix_path)
        for specific_task_path in glob(
            str(tokenizer_suffix_path) + "/*", recursive=True
        ):
            specific_task_path = Path(specific_task_path)
            specific_task = specific_task_path.name
            classification = (
                MOLNET_DIRECTORY[specific_task]["dataset_type"] == "classification"
            )
            if (specific_task_path / "checkpoint_best.pt").is_file():
                model = load_model(
                    specific_task_path / "checkpoint_best.pt", cuda_device
                )
                mols = load_dataset(specific_task_path / "input0" / "test")
                if classification:
                    labels = load_dataset(specific_task_path / "label" / "test")
                else:
                    labels = load_regression_dataset(
                        specific_task_path / "label" / "test.label"
                    )
                preds, seen_targets = get_predictions(
                    model,
                    mols,
                    labels,
                    specific_task_path / "label" / "dict.txt",
                    classification,
                )
                score_dict, report = get_score(preds, seen_targets, classification)
                if classification:
                    with open(specific_task_path / "report.txt", "w") as report_file:
                        report_file.write(report)
                else:
                    with open(specific_task_path / "report.txt", "w") as report_file:
                        report_file.write(json.dumps(report, indent=4))

                score_dict["task_type"] = (
                    "classification" if classification else "regression"
                )
                score_dict["task"] = specific_task
                score_dict["tokenizer"] = tokenizer_suffix_path.name
                score_dict["model"] = model_type
                pd.DataFrame([score_dict]).to_csv(specific_task_path / "scores.csv")
