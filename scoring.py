"""Scoring function to calculate custom scores of Fairseq models
SMILES or SELFIES, 2022
"""

import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from lexicographic_scores import compute_distances
from preprocessing import canonize_smile, translate_smile
from tqdm import tqdm

import pandas as pd
from constants import (
    MOLNET_DIRECTORY,
    TASK_MODEL_PATH,
    TASK_PATH,
    RETROSYNTHESIS_DIRECTORY,
    PROJECT_PATH,
)
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
from fairseq_utils import load_dataset, load_model, get_predictions
from utils import parse_arguments
import pandas as pd


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


def parse_line(line: str, separator_occurences=1):
    tokens = line.split("\t", separator_occurences)[separator_occurences]
    tokens = [token.strip() for token in tokens.split(" ") if token]
    unk_flag = "<unk>" in tokens
    full = "".join(tokens).strip()
    return full, unk_flag


def parse_file(file_path, examples_per=10):
    with open(file_path, "r") as open_file:
        lines = open_file.readlines()[:-1]
    samples = []
    assert (
        len(lines) % (2 + 3 * examples_per) == 0
    ), f"{len(lines)} does not work with examples per {examples_per}."
    target_examples = np.split(np.array(lines), len(lines) / (2 + 3 * examples_per))
    for target_example in tqdm(target_examples):
        sample_dict = {}
        source, source_unk = parse_line(target_example[0], 1)
        sample_dict["source"] = source
        sample_dict["source_unk"] = source_unk
        target, target_unk = parse_line(target_example[1], 1)
        sample_dict = sample_dict | compute_distances(source, target)
        sample_dict["target"] = target
        sample_dict["target_unk"] = target_unk
        predictions = []
        target_example = target_example[2:]
        for _ in range(examples_per):
            prediction, prediction_unk = parse_line(target_example[0], 2)
            predictions.append((prediction, prediction_unk))
            target_example = target_example[3:]
        sample_dict["predictions"] = predictions
        samples.append(sample_dict)
    return samples


def find_match(target, predictions, selfies):
    if selfies:
        canonized_target = translate_smile(target)
    else:
        canonized_target = canonize_smile(target)
    for index, prediction in enumerate(predictions):
        if selfies:
            canonized_prediction = translate_smile(prediction[0])
        else:
            canonized_prediction = canonize_smile(prediction[0])

        if (
            canonized_prediction is not None
            and canonized_target is not None
            and canonized_prediction == canonized_target
        ) or prediction == target:
            return index
    return None


def score_samples(samples, selfies=False):
    matches = [
        find_match(sample["target"], sample["predictions"], selfies)
        for sample in tqdm(samples)
    ]
    stats = {"all_samples": len(samples)}
    for i in range(len(samples[0]["predictions"])):
        stats[f"top_{i+1}"] = matches.count(i)
    return stats


def score_distances(samples):
    keep_keys = [
        "max_len",
        "len_diff",
        "nw",
        "nw_norm",
        "lev",
        "lev_norm",
        "dl",
        "dl_norm",
    ]
    df = [{key: sample[key] for key in keep_keys} for sample in samples]
    df = pd.DataFrame.from_dict(df)
    output = {}
    for key in keep_keys:
        output[f"{key}_mean"] = df[key].mean()
        output[f"{key}_median"] = df[key].median()
        output[f"{key}_std"] = df[key].std()
    return output


if __name__ == "__main__":
    cuda = parse_arguments(True, False, False)["cuda"]
    for task_path, task in iterate_paths(TASK_MODEL_PATH):
        for tokenizer_path, tokenizer in iterate_paths(task_path):
            if task in MOLNET_DIRECTORY:
                classification = (
                    MOLNET_DIRECTORY[task]["dataset_type"] == "classification"
                )
                for hyperparameter_path, hyperparameter in iterate_paths(
                    tokenizer_path
                ):
                    if (hyperparameter_path / hyperparameter).exists():
                        best_checkpoint_path = (
                            hyperparameter_path / hyperparameter / "checkpoint_best.pt"
                        )
                    else:
                        best_checkpoint_path = (
                            hyperparameter_path / "checkpoint_best.pt"
                        )
                    if not best_checkpoint_path.is_file():
                        continue
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
                        "tokenizer": tokenizer,
                        "task_type": "classification"
                        if classification
                        else "regression",
                    }
                    score_dict, report = get_score(preds, seen_targets, classification)
                    output_dict = (
                        output_dict | parse_hyperparams(hyperparameter) | score_dict
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
            if task in RETROSYNTHESIS_DIRECTORY:
                # os.system(
                #    f'CUDA_VISIBLE_DEVICES={cuda} fairseq-generate {TASK_PATH/task/tokenizer/"pre-processed"} --source-lang input --target-lang label --wandb-project retrosynthesis-beam-generate --task translation --path {TASK_MODEL_PATH/task/tokenizer/"1e-05_0.2_based_norm"/"checkpoint_best.pt"} --batch-size 16 --beam 10 --nbest 10 --results-path {PROJECT_PATH/"retrosynthesis_beam"/task/tokenizer}'
                # )
                samples = parse_file(
                    PROJECT_PATH
                    / "retrosynthesis_beam"
                    / task
                    / tokenizer
                    / "generate-test.txt"
                )
                selfies = "selfies" in tokenizer
                output = {"model": tokenizer, "task": task}
                output = output | score_samples(samples, selfies)
                output = output | score_distances(samples)
                pd.DataFrame.from_dict([output]).to_csv(
                    PROJECT_PATH
                    / "retrosynthesis_beam"
                    / task
                    / tokenizer
                    / "output.csv"
                )
