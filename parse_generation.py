""" Parse generation
SMILES or SELFIES 2023
"""
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import PROJECT_PATH, TASK_MODEL_PATH, TASK_PATH, TOKENIZER_SUFFIXES
from lexicographic_scores import compute_distances
from preprocessing import canonize_smile, translate_smile


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
        "rouge1",
        "rouge2",
        "rouge3",
        "rougeL",
        "BLEU",
        "BLEU1",
        "BLEU2",
        "BLEU3",
        "BLEU4",
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
    cuda = 3
    for task in ["lef", "jin", "schwaller"]:
        for tokenizer in TOKENIZER_SUFFIXES:
            # os.system(
            #    f'CUDA_VISIBLE_DEVICES={cuda} fairseq-generate {TASK_PATH/task/tokenizer/"reaction_prediction"} --source-lang input --target-lang label --wandb-project reaction_prediction-beam-generate --task translation --path {TASK_MODEL_PATH/task/tokenizer/"1e-05_0.2_based_norm"/"checkpoint_best.pt"} --batch-size 16 --beam 10 --nbest 10 --results-path {PROJECT_PATH/"reaction_prediction_beam"/task/tokenizer}'
            # )
            samples = parse_file(
                PROJECT_PATH
                / "reaction_prediction_beam"
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
                / "reaction_prediction_beam"
                / task
                / tokenizer
                / "output.csv"
            )
