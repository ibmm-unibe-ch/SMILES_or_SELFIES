"""Readout of attention
SMILES or SELFIES
"""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from constants import (
    MOLNET_DIRECTORY,
    PARSING_REGEX,
    PLOT_PATH,
    TASK_MODEL_PATH,
    TASK_PATH,
    TOKENIZER_PATH,
)
from deepchem.feat import RawFeaturizer
from fairseq_utils import compute_model_output
from plotting import plot_representations
from preprocessing import canonize_smile, create_identities, translate_selfie
from scoring import load_dataset, load_model
from tokenisation import get_tokenizer
from utils import log_and_add, parse_arguments

from fairseq.data import Dictionary


def find_attention_outliers(
    attention: pd.Series, quantile: float = 0.25, ratio: float = 1.5
):
    lower_quantile = attention.quantile(quantile)
    higher_quantile = attention.quantile(1 - quantile)
    inter_quartile_range = higher_quantile - lower_quantile
    return attention.gt(
        higher_quantile + ratio * inter_quartile_range
    )  # Series of bools


def aggregate_SMILE_attention(line: List[Tuple[float, str]]) -> dict:
    """Aggregate the attention from a SMILES line according to similar tokens
    http://www.dalkescientific.com/writings/diary/archive/2004/01/05/tokens.html

    Args:
        line (List[Tuple[float, str]]): Line consisting of attention score and token

    Returns:
        dict: dict with keys to indicate aggregation and scores
    """
    output_dict = {}
    bond_att = 0
    high_attention = find_attention_outliers(pd.Series([entry[0] for entry in line]))
    for counter, (score, token) in enumerate(line):
        output_dict[f"token count {token}"] = (
            output_dict.get(f"token count {token}", 0) + 1
        )
        if high_attention[counter]:
            output_dict[f"high count {token}"] = (
                output_dict.get(f"high count {token}", 0) + 1
            )
        if token in ["(", ")"] or token.isnumeric():
            output_dict["structure attention"] = (
                output_dict.get("structure attention", 0) + score
            )
            output_dict["structure attention trailing bond"] = (
                output_dict.get("structure attention trailing bond", 0)
                + score
                + bond_att
            )
            output_dict["structure count"] = output_dict.get("structure count", 0) + 1
            output_dict["structure count trailing bond"] = (
                output_dict.get("structure count trailing bond", 0) + 1
            )
            bond_att = 0
        elif token in ["=", "#", "/", "\\", ":", "~", "-"]:
            output_dict["bond attention"] = output_dict.get("bond attention", 0) + score
            output_dict["bond count"] = output_dict.get("bond count", 0) + 1
            bond_att += score
        else:
            output_dict["atom attention"] = output_dict.get("atom attention", 0) + score
            output_dict["atom attention trailing bond"] = (
                output_dict.get("atom attention trailing bond", 0) + score + bond_att
            )
            output_dict["atom count"] = output_dict.get("atom count", 0) + 1
            output_dict["atom count trailing bond"] = (
                output_dict.get("atom count trailing bond", 0) + 1
            )
            bond_att = 0
    # distribute bond attention
    output_dict["structure attention distributed"] = output_dict.get(
        "structure attention", 0
    ) + output_dict.get("bond attention", 0) * (
        output_dict.get("structure attention", 0)
        / (
            output_dict.get("structure attention", 0)
            + output_dict.get("atom attention", 0)
        )
    )
    output_dict["atom attention distributed"] = output_dict[
        "atom attention"
    ] + output_dict.get("bond attention", 0) * (
        output_dict["atom attention"]
        / (output_dict.get("structure attention", 0) + output_dict["atom attention"])
    )
    output_dict["structure count distributed"] = output_dict.get(
        "structure count", 0
    ) + output_dict.get("bond count", 0) * (
        output_dict.get("structure count", 0)
        / (output_dict.get("structure count", 0) + output_dict.get("atom count", 0))
    )
    output_dict["atom count distributed"] = output_dict.get(
        "atom count", 0
    ) + output_dict.get("bond count", 0) * (
        output_dict.get("atom count", 0)
        / (output_dict.get("structure count", 0) + output_dict.get("atom count", 0))
    )
    return output_dict


def aggregate_SELFIE_attention(line: List[Tuple[float, str]]) -> dict:
    """Aggregate the attention from a SELFIES line according to similar tokens

    Args:
        line (List[Tuple[float, str]]): Line consisting of attention score and token

    Returns:
        dict: dict with keys to indicate aggregation and scores
    """
    output_dict = {}
    structure_tokens = 0
    high_attention = find_attention_outliers(pd.Series([entry[0] for entry in line]))
    remaining_structure_tokens = 0
    for counter, (score, token) in enumerate(line):
        if structure_tokens > 0:
            noting_token = str(token) + " overloaded"
        else:
            noting_token = token
        if high_attention[counter]:
            output_dict[f"high count {noting_token}"] = (
                output_dict.get(f"high count {noting_token}", 0) + 1
            )
        output_dict[f"token count {noting_token}"] = (
            output_dict.get(f"token count {noting_token}", 0) + 1
        )
        if remaining_structure_tokens > 0:
            # overloaded tokens
            remaining_structure_tokens -= 1
            structure_score += score
        elif "Ring" in token or "Branch" in token:
            # parse overloading
            output_dict["structure count"] = output_dict.get("structure count", 0) + 1
            structure_tokens = int(token[-2]) + 1
            # do not need Ring/Branch token
            remaining_structure_tokens = structure_tokens - 1
            structure_score = score
        else:
            output_dict["atom attention"] = output_dict.get("atom attention", 0) + score
            output_dict["atom count"] = output_dict.get("atom count", 0) + 1
        if remaining_structure_tokens == 0 and structure_tokens > 0:
            output_dict["structure attention"] = (
                output_dict.get("structure attention", 0)
                + structure_score / structure_tokens
            )
            structure_tokens = 0
            structure_score = 0
    return output_dict


def aggregate_SMILE_embeddings(line: List[Tuple[float, str]]) -> dict:
    """Aggregate the attention from a SMILES line according to similar tokens
    http://www.dalkescientific.com/writings/diary/archive/2004/01/05/tokens.html

    Args:
        line (List[Tuple[float, str]]): Line consisting of attention score and token

    Returns:
        count_dict, embedding_dict
    """
    count_dict = {}
    embedding_dict = {}
    for embedding, token in line:
        count_dict[token] = count_dict.get(token, 0) + 1
        embedding_dict[token] = embedding_dict.get(
            token, np.zeros(len(embedding))
        ) + np.array(embedding)
    return count_dict, embedding_dict


def aggregate_SELFIE_embeddings(line: List[Tuple[float, str]]) -> Tuple[dict, dict]:
    """Aggregate the embeddings from a SELFIES line according to similar tokens

    Args:
        line (List[Tuple[float, str]]): Line consisting of attention score and token

    Returns:
        count_dict, embedding_dict
    """
    count_dict = {}
    embedding_dict = {}
    remaining_structure_tokens = 0
    for (embedding, token) in line:
        if remaining_structure_tokens > 0:
            noting_token = str(token) + " overloaded"
        else:
            noting_token = token
        count_dict[noting_token] = count_dict.get(noting_token, 0) + 1
        embedding_dict[noting_token] = embedding_dict.get(
            noting_token, np.zeros(len(embedding))
        ) + np.array(embedding)
        if remaining_structure_tokens > 0:
            # overloaded tokens
            remaining_structure_tokens -= 1
        elif "Ring" in token or "Branch" in token:
            # parse overloading
            remaining_structure_tokens = int(token[-2])
    return count_dict, embedding_dict


def parse_att_dict(
    SMILE_dict: dict, SELFIE_dict: dict, len_output: int, save_path: Path
):
    """Parse the attention dicts.

    Args:
        SMILE_dict (dict): dict of SMILES
        SELFIE_dict (dict): dict of SELFIES
        len_output (int): amount of samples
        save_path (Path): path to save aggregates to
    """
    text = ""
    for (representation, dikt) in [("SMILES", SMILE_dict), ("SELFIES", SELFIE_dict)]:
        text += log_and_add(text, f"{representation}:")
        for key in sorted(dikt.keys()):
            if ("token" in key) or ("high" in key):
                continue
            text = log_and_add(text, f"The amount of {key} is {dikt[key]:.3f}.")
            if "attention" in key:
                modified_key = key.replace("attention", "count")
                text = log_and_add(
                    text,
                    f"The amount of {key} per token is {dikt[key]/dikt[modified_key]:.3f}",
                )
            if "high" in key:
                token = key[len("high count ") :]
                token_count_key = "token count " + token
                text = log_and_add(
                    text,
                    f"The percentage of high attention token: {token} is {dikt[key]/dikt[token_count_key]:.3f}",
                )
                text = log_and_add(
                    text, f"The amount of {token} is {dikt[token_count_key]}"
                )
                continue
            text = log_and_add(
                text, f"The amount of {key} per sample is {dikt[key]/len_output:.3f}"
            )
    text = log_and_add(text, str(SMILE_dict))
    text = log_and_add(text, str(SELFIE_dict))
    with open(save_path, "w") as openfile:
        openfile.write(text)


def load_molnet_test_set(task: str) -> Tuple[List[str], List[int]]:
    """Load MoleculeNet task

    Args:
        task (str): MoleculeNet task to load

    Returns:
        Tuple[List[str], List[int]]: Features, Labels
    """
    task_test = MOLNET_DIRECTORY[task]["load_fn"](
        reload=False,
        featurizer=RawFeaturizer(smiles=True),
        splitter=MOLNET_DIRECTORY[task]["split"],
    )[1][2]
    task_SMILES = task_test.X
    task_labels = task_test.y
    return task_SMILES, task_labels


def generate_output_dict(task: str, cuda: int) -> Tuple[List[List[float]], np.ndarray]:
    """Generate the attention dict of a task

    Args:
        task (str): Task to find attention of
        cuda (int): CUDA device to use

    Returns:
        Tuple[List[List[float]], np.ndarray]: attention, labels
    """
    task_SMILES, task_labels = load_molnet_test_set(task)
    all_attention = []
    all_embeddings = []
    specific_model_params = {
        "smiles_atom": "5e-05_0.2_based_norm",
        "selfies_atom": "5e-05_0.2_based_norm",
        "smiles_sentencepiece": "5e-05_0.1_based_norm",
        "selfies_sentencepiece": "5e-05_0.3_based_norm",
    }
    for encoding in [
        "smiles_atom",
        "selfies_atom",
        "smiles_sentencepiece",
        "selfies_sentencepiece",
    ]:
        specific_model_param = specific_model_params[encoding]
        specific_model_path = (
            TASK_MODEL_PATH
            / task
            / encoding
            / specific_model_param
            / specific_model_param
            / "checkpoint_best.pt"
        )
        data_path = TASK_PATH / task / encoding
        model = load_model(specific_model_path, data_path, cuda)
        model.zero_grad()
        data_path = data_path / "input0" / "test"
        dataset = load_dataset(data_path)
        source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))
        assert len(task_SMILES) == len(
            dataset
        ), f"Real and filtered dataset {task} do not have same length."
        text = [canonize_smile(smile) for smile in task_SMILES]
        if encoding.startswith("selfies"):
            text = [translate_selfie(smile)[0] for smile in text]

        if encoding.endswith("sentencepiece"):
            tokenizer = get_tokenizer(TOKENIZER_PATH / encoding)
        else:
            tokenizer = None
        _, attention, embeddings, _ = compute_model_output(
            dataset,
            model,
            text,
            source_dictionary,
            False,
            True,
            True,
            False,
            tokenizer,
        )
        all_attention.append(attention)
        all_embeddings.append(embeddings)
    all_attention = list(zip(*all_attention))
    all_embeddings = list(zip(*all_embeddings))
    labels = np.array(task_labels).transpose()[0]
    return all_attention, all_embeddings, labels


def aggregate_attention(output: List[Tuple[float, str]]) -> Tuple[dict, dict]:
    """Aggregate the attention of annotated

    Args:
        output (str): List of attention and token

    Returns:
        Tuple[dict,dict]: SMILES_dict, SELFIES_dict
    """
    SMILE_dict = {}
    SELFIE_dict = {}
    for line in output:
        curr_dict = aggregate_SMILE_attention(line[0])
        SMILE_dict = {
            key: SMILE_dict.get(key, 0) + curr_dict.get(key, 0)
            for key in SMILE_dict | curr_dict
        }
        curr_dict = aggregate_SELFIE_attention(line[1])
        SELFIE_dict = {
            key: SELFIE_dict.get(key, 0) + curr_dict.get(key, 0)
            for key in SELFIE_dict | curr_dict
        }
    return SMILE_dict, SELFIE_dict


def aggregate_embeddings(all_embeddings):
    SMILES_count_dict = {}
    SELFIES_count_dict = {}
    SMILES_embedding_dict = {}
    SELFIES_embedding_dict = {}
    for line in all_embeddings:
        count_dict, embedding_dict = aggregate_SMILE_embeddings(line[0])
        SMILES_count_dict = {
            key: SMILES_count_dict.get(key, 0) + count_dict.get(key, 0)
            for key in SMILES_count_dict | count_dict
        }
        SMILES_embedding_dict = {
            key: SMILES_embedding_dict.get(key, np.zeros(len(line[0][0][0])))
            + embedding_dict.get(key, np.zeros(len(line[0][0][0])))
            for key in SMILES_embedding_dict | embedding_dict
        }
        count_dict, embedding_dict = aggregate_SELFIE_embeddings(line[1])
        SELFIES_count_dict = {
            key: SELFIES_count_dict.get(key, 0) + count_dict.get(key, 0)
            for key in SELFIES_count_dict | count_dict
        }
        SELFIES_embedding_dict = {
            key: SELFIES_embedding_dict.get(key, np.zeros(len(line[1][0][0])))
            + embedding_dict.get(key, np.zeros(len(line[1][0][0])))
            for key in SELFIES_embedding_dict | embedding_dict
        }

    SMILES_dict = {
        key: SMILES_embedding_dict[key] / SMILES_count_dict[key]
        for key in SMILES_count_dict
    }
    SELFIES_dict = {
        key: SELFIES_embedding_dict[key] / SELFIES_count_dict[key]
        for key in SELFIES_count_dict
    }
    return SMILES_dict, SELFIES_dict


def produce_att_samples(
    output: List[Tuple[float, str]],
    labels: List[int],
    save_dir: Path,
    samples: int = 20,
):
    """Produce *samples* many samples from output with labels and save them to save_dir

    Args:
        output (List[Tuple[float, str]]): List of attention and token
        labels (List[int]): Labels of samples
        save_dir (Path): where to save samples to
        samples (int, optional): amount of samples to select. Defaults to 20.
    """
    md = ""
    for i in range(samples):
        md += f"# Sample {i+1} with value {labels[i]:.3f}" + r"\n"
        md += "## Molecule" + r"\n"
        input_data = np.array([letter[1] for letter in output[i][0]])
        md += f'{"".join(input_data)}' + r"\n"
        md += (
            f'{[token for token in re.split(PARSING_REGEX,create_identities("".join(input_data))[0]) if token] }'
            + r"\n"
        )
        for it, tokenizer in enumerate(
            [
                "SMILES",
                "SELFIES",
            ]  # "SMILES SentencePiece", "SMILES SentencePiece"]
        ):
            md += f"## {tokenizer}" + r"\n"
            outliers = find_attention_outliers(
                pd.Series([token[0] for token in output[i][it]])
            )
            input_data_corrected = np.array(
                [
                    (
                        "*" + str(letter[1]) + "*"
                        if outliers[counter]
                        else str(letter[1]),
                        f"{letter[0]-1/len(output[i][it]):.3f}",
                    )
                    for counter, letter in enumerate(output[i][it])
                ]
            )
            input_data_pure = np.array(
                [
                    (
                        "*" + str(letter[1]) + "*"
                        if outliers[counter]
                        else str(letter[1]),
                        f"{letter[0]:.3f}",
                    )
                    for counter, letter in enumerate(output[i][it])
                ]
            )
            df = pd.concat(
                [
                    pd.DataFrame(
                        data=input_data_corrected[:, 1],
                        index=input_data_corrected[:, 0],
                    )
                    .transpose()
                    .reset_index(drop=True),
                    pd.DataFrame(
                        data=input_data_pure[:, 1], index=input_data_pure[:, 0]
                    )
                    .transpose()
                    .reset_index(drop=True),
                ]
            )
            md += df.to_markdown() + r"\n"
    with open(save_dir, "w") as openfile:
        openfile.write(md)


if __name__ == "__main__":
    args = parse_arguments(True, False, True)
    assert args["task"] in list(
        MOLNET_DIRECTORY.keys()
    ), f"{args['task']} not in MOLNET tasks."
    all_attention, all_embeddings, labels = generate_output_dict(
        args["task"], args["cuda"]
    )
    SMILES_embedding_dict, SELFIES_embedding_dict = aggregate_embeddings(all_embeddings)
    plot_representations(
        list(SMILES_embedding_dict.values()),
        pd.Series(list(SMILES_embedding_dict.keys())),
        PLOT_PATH / "SMILES_aggregated_tokens",
    )
    plot_representations(
        list(SELFIES_embedding_dict.values()),
        pd.Series(
            [
                "Overloaded" if "overloaded" in element else "Standard"
                for element in list(SELFIES_embedding_dict.keys())
            ]
        ),
        PLOT_PATH / "SELFIES_aggregated_tokens",
    )
    SMILE_dict, SELFIE_dict = aggregate_attention(all_attention)
    parse_att_dict(
        SMILE_dict,
        SELFIE_dict,
        len(all_attention),
        f"logs/attention_agg_{args['task']}.txt",
    )
    produce_att_samples(
        all_attention, labels, f"logs/attention_samples_{args['task']}.md"
    )
