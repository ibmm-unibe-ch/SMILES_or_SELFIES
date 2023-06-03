""" Parse generation file 
SMILES or SELFIES 2023
"""
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import PROJECT_PATH, TASK_MODEL_PATH, TASK_PATH, TOKENIZER_SUFFIXES
from lexicographic_scores import compute_distances
from preprocessing import canonize_smile, check_valid, translate_smile
from utils import parse_arguments


def parse_line(line: str, separator_occurences: int = 1) -> Tuple[str, int]:
    """Parse line after seperator_occurences of the seperator

    Args:
        line (str): line to parse
        separator_occurences (int, optional): how many seperator was seen before the good sequence. Defaults to 1.

    Returns:
        Tuple[str, bool, int]: parsed mol, amount of NLP tokens
    """
    tokens = line.split("\t", separator_occurences)[separator_occurences]
    tokens = [token.strip() for token in tokens.split(" ") if token]
    full = "".join(tokens).strip()
    return full, len(tokens)


def parse_file(file_path: Path, examples_per: int = 10) -> dict:
    """Parsing of generation file made by fairseq-generate

    Args:
        file_path (Path): Path to fairseq-generated file
        examples_per (int, optional): How many examples . Defaults to 10.

    Returns:
        dict: _description_
    """
    with open(file_path, "r") as open_file:
        lines = open_file.readlines()[:-1]
    samples = []
    assert (
        len(lines) % (2 + 3 * examples_per) == 0
    ), f"{len(lines)} does not work with examples per {examples_per}."
    target_examples = np.split(np.array(lines), len(lines) / (2 + 3 * examples_per))
    for target_example in tqdm(target_examples):
        sample_dict = {}
        source, source_len = parse_line(target_example[0], 1)
        sample_dict["source"] = source
        sample_dict["source_len"] = source_len
        target, target_len = parse_line(target_example[1], 1)
        sample_dict = sample_dict | compute_distances(source, target)
        sample_dict["target"] = target
        sample_dict["target_len"] = target_len
        predictions = []
        target_example = target_example[2:]
        for _ in range(examples_per):
            prediction, _ = parse_line(target_example[0], 2)
            predictions.append(prediction)
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
    stats = {"all_samples": len(samples)}
    samples = pd.DataFrame(samples)
    samples["matches"] = samples.apply(
        lambda row: find_match(row["target"], row["predictions"], selfies),
        axis="columns",
    )
    prediction_df = pd.DataFrame(
        samples["predictions"].to_list(),
        columns=[
            f"prediction_{i+1}" for i in range(len(samples.loc[0]["predictions"]))
        ],
    )
    rank_counts = samples["matches"].value_counts()
    for rank in rank_counts.index:
        stats[f"top_{int(float(str(rank).strip()))+1}"] = rank_counts[rank]
    for column in prediction_df.columns:
        stats[f'unk_{column[len("prediction_"):]}'] = sum(
            prediction_df[column].str.contains("unk")
        )
        stats[f'valid_{column[len("prediction_"):]}'] = sum(
            prediction_df[column].apply(
                lambda x: not (translate_smile(x) is None)
                if selfies
                else check_valid(x)
            )
        )
    return stats


def score_distances(samples):
    keep_keys = [
        "source_len",
        "target_len",
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
        "input_set",
        "output_set",
    ]
    df = [{key: sample[key] for key in keep_keys} for sample in samples]
    df = pd.DataFrame.from_dict(df)
    output = {}
    for key in keep_keys[:-2]:
        output[f"{key}_mean"] = df[key].mean()
        output[f"{key}_median"] = df[key].median()
        output[f"{key}_std"] = df[key].std()
    output["unique_input_tokens"] = len(set.union(*df["input_set"].tolist()))
    output["unique_output_tokens"] = len(set.union(*df["output_set"].tolist()))
    return output


if __name__ == "__main__":
    cuda = parse_arguments(True, False, False)["cuda"]
    for task in ["lef"]:  # , "jin", "schwaller"]:
        tokenizer = "smiles_isomers_atom"
        # for it, tokenizer in ["smiles_atom"]:#enumerate(TOKENIZER_SUFFIXES):
        best_models = [
            "0.0001_0.25_large_norm", "5e-05_0.25_large_norm"
        ]
        for it, model in enumerate(best_models):
            os.system(
                f'CUDA_VISIBLE_DEVICES={cuda} fairseq-generate {TASK_PATH/task/tokenizer/"reaction_prediction"} --source-lang input --target-lang label --wandb-project reaction_prediction-beam-regulation --task translation --path /ibmm_data/jgut/big_SoS_models/{task}/{tokenizer}/{model}/checkpoint_best.pt --batch-size 1 --skip-invalid-size-inputs-valid-test --beam 10 --nbest 10 --results-path {PROJECT_PATH/"reaction_prediction_beam_reg"/task/tokenizer/model}'
            )
            samples = parse_file(
                PROJECT_PATH
                / "reaction_prediction_beam_reg"
                / task
                / tokenizer
                / model
                / "generate-test.txt"
            )
            selfies = "selfies" in tokenizer
            output = {"model": tokenizer, "task": task}
            output = output | score_samples(samples, selfies)
            output = output | score_distances(samples)
            os.makedirs(
                PROJECT_PATH
                / "reaction_prediction_beam_reg"
                / task
                / tokenizer
                / model,
                exist_ok=True,
            )
            pd.DataFrame.from_dict([output]).to_csv(
                PROJECT_PATH
                / "reaction_prediction_beam_reg"
                / task
                / tokenizer
                / model
                / "output.csv"
            )
