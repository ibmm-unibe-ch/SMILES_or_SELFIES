"""Scoring function to calculate custom scores of Fairseq models
SMILES or SELFIES, 2022
"""

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
    auc,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from tqdm import tqdm

from constants import TASK_PATH


def load_model(model_path: Path, cuda_device: str = None):
    model = BARTModel.from_pretrained(
        model_path.parent, checkpoint_file=model_path.name
    )
    model.eval()
    if cuda_device:
        model.cuda(device=cuda_device)
    return model


def load_dataset(data_path: Path) -> np.ndarray:
    dikt = Dictionary.load(data_path.parent / "dict.txt")
    data = list(load_indexed_dataset(str(data_path), dikt))
    return data


def get_predictions(
    model, mols: np.ndarray, targets: np.ndarray, target_dict: Dict
) -> Tuple[List[float], List[float]]:
    preds = []
    seen_targets = []
    for _, (smile, target) in tqdm(list(enumerate(zip(mols, targets)))):
        smile = torch.cat(
            (torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2]))
        )
        output = model.predict(
            "sentence_classification_head", smile, return_logits=False
        )
        target = target[0].item()
        if target_dict.__getitem__(4) == "1":
            preds.append(output[0][0].exp().item())
            seen_targets.append(-1 * target + 5)
        else:
            preds.append(output[0][1].exp().item())
            seen_targets.append(target - 4)
    return preds, seen_targets


def get_score(predictions: List[float], seen_targets: List[float]) -> Tuple[dict, str]:
    score_dikt = {}
    roc_auc = roc_auc_score(seen_targets, predictions)
    score_dikt["ROC_AUC"] = roc_auc
    average_precision = average_precision_score(seen_targets, predictions)
    score_dikt["average_precision"] = average_precision
    auc_score = auc(seen_targets, predictions)
    score_dikt["AUC"] = auc_score
    acc_score = accuracy_score(seen_targets, predictions)
    score_dikt["accuracy_score"] = acc_score
    report = classification_report(seen_targets, predictions)
    return score_dikt, report


if __name__ == "__main__":
    cuda_device = 3
    for tokenizer_suffix_path in glob(str(TASK_PATH) + "/*", recursive=True):
        tokenizer_suffix_path = Path(tokenizer_suffix_path)
        for specific_task_path in glob(
            str(tokenizer_suffix_path) + "/*", recursive=True
        ):
            specific_task_path = Path(specific_task_path)
            if (specific_task_path / "checkpoint_best.pt").exists():
                model = load_model(
                    specific_task_path / "checkpoint_best.pt", cuda_device
                )
                mols = load_dataset(specific_task_path / "input0" / "test")
                labels = load_dataset(specific_task_path / "label" / "test")
                preds, seen_targets = get_predictions(
                    model, mols, labels, specific_task_path / "dict.txt"
                )
                score_dict, report = get_score(preds, seen_targets)
                with open(specific_task_path / "report.txt", "w") as report_file:
                    report_file.write(report)
                score_dict["task"] = specific_task_path.name
                score_dict["tokenizer"] = tokenizer_suffix_path.name
                score_dict["model"] = "base"
                pd.DataFrame(score_dict).to_csv(specific_task_path / "scores.csv")
