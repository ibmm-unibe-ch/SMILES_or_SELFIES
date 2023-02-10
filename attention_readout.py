import logging
import re

import numpy as np
import pandas as pd
from deepchem.feat import RawFeaturizer
from fairseq.data import Dictionary

from constants import (
    MOLNET_DIRECTORY,
    PARSING_REGEX,
    TASK_MODEL_PATH,
    TASK_PATH,
    TOKENIZER_PATH,
)
from preprocessing import canonize_smile, translate_selfie
from scoring import load_dataset, load_model
from tokenisation import get_tokenizer
from utils import parse_arguments


def generate_prev_output_tokens(sample, source_dictionary):
    tokens = sample.unsqueeze(-1)
    prev_output_tokens = tokens.clone()
    prev_output_tokens[:, 0] = tokens.gather(
        0, (tokens.ne(source_dictionary.pad()).sum(0) - 1).unsqueeze(-1)
    ).squeeze()
    prev_output_tokens[:, 1:] = tokens[:, :-1]
    return prev_output_tokens


def compute_attention_output(dataset, model, text, source_dictionary, tokenizer=None):
    # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py
    device = next(model.parameters()).device
    attentions = []
    for counter, sample in enumerate(dataset):
        if tokenizer is None:
            parsed_tokens = [
                parsed_token
                for parsed_token in re.split(PARSING_REGEX, text[counter])
                if parsed_token
            ]
        else:
            parsed_tokens = tokenizer.convert_ids_to_tokens(
                tokenizer(str(text[counter])).input_ids
            )
        prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(
            device
        )
        # same as in predict
        attention = model.model(
            sample.unsqueeze(0).to(device), None, prev_output_tokens
        )[1]["attn"][0][0][0].tolist()
        attentions.append(list(zip(attention, parsed_tokens)))
    return attentions


def aggregate_SMILE_attention(line):
    # http://www.dalkescientific.com/writings/diary/archive/2004/01/05/tokens.html
    output_dict = {}
    bond = ""
    bond_att = 0
    additional_attention = 0
    for (score, token) in line:
        if token in ["C", "c"]:
            output_dict[f"{bond}C count"] = output_dict.get(f"{bond}C count", 0) + 1
            output_dict[f"{bond}C attention"] = (
                output_dict.get(f"{bond}C attention", 0) + score + bond_att
            ) + additional_attention
            bond = ""
            bond_att = 0
        if token in ["(", ")"] or token.isnumeric():
            output_dict["structure attention"] = (
                output_dict.get("structure attention", 0) + score
            )
            output_dict["structure attenton trailing bond"] = (
                output_dict.get("structure attenton trailing bond", 0)
                + score
                + additional_attention
            )
            output_dict["structure count"] = output_dict.get("structure count", 0) + 1
            additional_attention = 0
        elif token in ["=", "#", "/", "\\", ":", "~", "-"]:
            output_dict["bond attention"] = output_dict.get("bond attention", 0) + score
            output_dict["bond count"] = output_dict.get("bond count", 0) + 1
            bond = token
            bond_att += score
            additional_attention += score
        else:
            output_dict["atom attention"] = output_dict.get("atom attention", 0) + score
            output_dict["atom attention trailing bond"] = (
                output_dict.get("atom attention trailing bond", 0)
                + score
                + additional_attention
            )
            output_dict["atom count"] = output_dict.get("atom count", 0) + 1
            output_dict["atom count trailing bond"] = (
                output_dict.get("atom count trailing bond", 0) + 1
            )
            additional_attention = 0
    # distribute bond attention
    output_dict["structure attention distributed"] = output_dict.get(
        "structure attention distributed", 0
    ) + output_dict.get("bond attention", 0) * (
        output_dict.get("structure attention", 0)
        / (
            output_dict.get("structure attention", 0)
            + output_dict.get("bond attention", 0)
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
        / (output_dict.get("structure count", 0) + output_dict.get("bond count", 0))
    )
    output_dict["atom count distributed"] = output_dict.get(
        "atom count", 0
    ) + output_dict.get("bond count", 0) * (
        output_dict.get("atom count", 0)
        / (output_dict.get("structure count", 0) + output_dict.get("atom count", 0))
    )
    return output_dict


def aggregate_SELFIE_attention(line):
    output_dict = {}
    structure_tokens = 0
    for (score, token) in line:
        if structure_tokens > 0:
            # overloaded tokens
            structure_tokens -= 1
            output_dict["structure attention"] = (
                output_dict.get("structure attention", 0) + score
            )
            output_dict["structure count"] = output_dict.get("structure count", 0) + 1
        elif "Ring" in token or "Branch" in token:
            output_dict["structure attention"] = (
                output_dict.get("structure attention", 0) + score
            )
            output_dict["structure count"] = output_dict.get("structure count", 0) + 1
            # parse overloading
            structure_tokens = int(token[-2])
        else:
            output_dict["atom attention"] = output_dict.get("atom attention", 0) + score
            output_dict["atom count"] = output_dict.get("atom count", 0) + 1
            if "C]" in token:
                if "C" == token[1]:
                    token = list(token)
                    token[1] = ""
                output_dict[f"{token[1]}C count"] = (
                    output_dict.get(f"{token[1]}C count", 0) + 1
                )
                output_dict[f"{token[1]}C attention"] = (
                    output_dict.get(f"{token[1]}C attention", 0) + score
                )
    return output_dict


def log_and_add(text, string):
    logging.info(string)
    text += string + "\n"
    return text


def parse_att_dict(SMILE_dict, SELFIE_dict, len_output, save_path):
    text = ""
    for (representation, dikt) in [("SMILES", SMILE_dict), ("SELFIES", SELFIE_dict)]:
        text += log_and_add(text, f"{representation}:")
        for key in sorted(dikt.keys()):
            text = log_and_add(text, f"The amount of {key} is {dikt[key]:.3f}.")
            if "attention" in key:
                modified_key = key.replace("attention", "count")
                text = log_and_add(
                    text,
                    f"The amount of {key} per token is {dikt[key]/dikt[modified_key]:.3f}",
                )
            text = log_and_add(
                text, f"The amount of {key} per sample is {dikt[key]/len_output:.3f}"
            )
    text = log_and_add(text, str(SMILE_dict))
    text = log_and_add(text, str(SELFIE_dict))
    with open(save_path, "w") as openfile:
        openfile.write(text)


def load_molnet_test_set(task):
    task_test = MOLNET_DIRECTORY[task]["load_fn"](
        featurizer=RawFeaturizer(smiles=True), splitter=MOLNET_DIRECTORY[task]["split"]
    )[1][2]
    task_SMILES = task_test.X
    task_labels = task_test.y
    return task_SMILES, task_labels


def generate_attention_dict(task, cuda):
    task_SMILES, task_labels = load_molnet_test_set(task)
    for encoding in [
        "smiles_atom",
        "selfies_atom",
        "smiles_sentencepiece",
        "selfies_sentencepiece",
    ]:
        specific_model_path = (
            TASK_MODEL_PATH
            / task
            / encoding
            / "5e-05_0.2_based_norm"
            / "5e-05_0.2_based_norm"
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
        attention_encodings.append(
            compute_attention_output(dataset, model, text, source_dictionary, tokenizer)
        )
    output = list(zip(*attention_encodings))
    labels = np.array(task_labels).transpose()[0]
    return output, labels


def aggregate_attention(output):
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


def produce_att_samples(output, labels, save_dir, samples=20):
    md = ""
    for i in range(samples):
        md += f"# Sample {i+1} with value {labels[i]:.3f}" + r"\n"
        md += "## Molecule" + r"\n"
        input_data = np.array(
            [
                (letter[1], f"{letter[0]-1/len(output[i][0]):.3f}")
                for letter in output[i][0]
            ]
        )
        md += f'{"".join(input_data[:,0])}' + r"\n"
        for it, tokenizer in enumerate(
            ["SMILES", "SELFIES", "SMILES SentencePiece", "SMILES SentencePiece"]
        ):
            md += f"## {tokenizer}" + r"\n"
            input_data_corrected = np.array(
                [
                    (letter[1], f"{letter[0]-1/len(output[i][it]):.3f}")
                    for letter in output[i][it]
                ]
            )
            input_data_pure = np.array(
                [(letter[1], f"{letter[0]:.3f}") for letter in output[i][it]]
            )
            df = pd.concat(
                [
                    pd.DataFrame(
                        data=input_data_corrected[:, 1],
                        index=input_data_corrected[:, 0],
                    ).transpose(),
                    pd.DataFrame(
                        data=input_data_pure[:, 1], index=input_data_pure[:, 0]
                    ).transpose(),
                ]
            )
            md += df.to_markdown() + r"\n"
    with open(save_dir, "w") as openfile:
        openfile.write(md)


if __name__ == "__main__":
    attention_encodings = []
    args = parse_arguments(True, False, True)
    assert args["task"] in list(
        MOLNET_DIRECTORY.keys()
    ), f"{args['task']} not in MOLNET tasks."
    output, labels = generate_attention_dict(args["task"], args["cuda"])
    SMILE_dict, SELFIE_dict = aggregate_attention(output)
    parse_att_dict(
        SMILE_dict, SELFIE_dict, len(output), f"logs/attention_agg_{args['task']}.txt"
    )
    produce_att_samples(output, labels, f"logs/attention_samples_{args['task']}.md")
