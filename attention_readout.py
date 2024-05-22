"""Readout of attention
SMILES or SELFIES
"""

#import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from constants import FAIRSEQ_PREPROCESS_PATH, PREDICTION_MODEL_PATH, TOKENIZER_PATH, TASK_PATH, MOLNET_DIRECTORY#(
#    MOLNET_DIRECTORY,
#    PARSING_REGEX,
#    TASK_MODEL_PATH,
#    TASK_PATH,
#    TOKENIZER_PATH,
#)
from deepchem.feat import RawFeaturizer
from fairseq_utils import compute_model_output, compute_attention_output, get_dictionary
from preprocessing import canonize_smile, create_identities, translate_selfie
from scoring import load_model#, load_dataset
from tokenisation import get_tokenizer
#from utils import log_and_add, parse_arguments

#from fairseq.data import Dictionary
#
#
#def find_attention_outliers(
#    attention: pd.Series, quantile: float = 0.25, ratio: float = 1.5
#):
#    lower_quantile = attention.quantile(quantile)
#    higher_quantile = attention.quantile(1 - quantile)
#    inter_quartile_range = higher_quantile - lower_quantile
#    return attention.gt(
#        higher_quantile + ratio * inter_quartile_range
#    )  # Series of bools
#
#
#def aggregate_SMILE_attention(line: List[Tuple[float, str]]) -> dict:
#    """Aggregate the attention from a SMILES line according to similar tokens
#    http://www.dalkescientific.com/writings/diary/archive/2004/01/05/tokens.html
#
#    Args:
#        line (List[Tuple[float, str]]): Line consisting of attention score and token
#
#    Returns:
#        dict: dict with keys to indicate aggregation and scores
#    """
#    output_dict = {}
#    # bond = ""
#    bond_att = 0
#    high_attention = find_attention_outliers(pd.Series([entry[0] for entry in line]))
#    for counter, (score, token) in enumerate(line):
#        output_dict[f"token count {token}"] = (
#            output_dict.get(f"token count {token}", 0) + 1
#        )
#        if high_attention[counter]:
#            output_dict[f"high count {token}"] = (
#                output_dict.get(f"high count {token}", 0) + 1
#            )
#        # if token in ["C", "c"]:
#        #    output_dict[f"{bond}C count"] = output_dict.get(f"{bond}C count", 0) + 1
#        #    output_dict[f"{bond}C attention"] = (
#        #        output_dict.get(f"{bond}C attention", 0) + score + bond_att
#        #    )
#        #    bond = ""
#        #    bond_att = 0
#        if token in ["(", ")"] or token.isnumeric():
#            output_dict["structure attention"] = (
#                output_dict.get("structure attention", 0) + score
#            )
#            output_dict["structure attention trailing bond"] = (
#                output_dict.get("structure attention trailing bond", 0)
#                + score
#                + bond_att
#            )
#            output_dict["structure count"] = output_dict.get("structure count", 0) + 1
#            output_dict["structure count trailing bond"] = (
#                output_dict.get("structure count trailing bond", 0) + 1
#            )
#            bond_att = 0
#            # bond = ""
#        elif token in ["=", "#", "/", "\\", ":", "~", "-"]:
#            output_dict["bond attention"] = output_dict.get("bond attention", 0) + score
#            output_dict["bond count"] = output_dict.get("bond count", 0) + 1
#            # bond = token
#            bond_att += score
#        else:
#            output_dict["atom attention"] = output_dict.get("atom attention", 0) + score
#            output_dict["atom attention trailing bond"] = (
#                output_dict.get("atom attention trailing bond", 0) + score + bond_att
#            )
#            output_dict["atom count"] = output_dict.get("atom count", 0) + 1
#            output_dict["atom count trailing bond"] = (
#                output_dict.get("atom count trailing bond", 0) + 1
#            )
#            bond_att = 0
#            # bond = ""
#    # distribute bond attention
#    output_dict["structure attention distributed"] = output_dict.get(
#        "structure attention", 0
#    ) + output_dict.get("bond attention", 0) * (
#        output_dict.get("structure attention", 0)
#        / (
#            output_dict.get("structure attention", 0)
#            + output_dict.get("atom attention", 0)
#        )
#    )
#    output_dict["atom attention distributed"] = output_dict[
#        "atom attention"
#    ] + output_dict.get("bond attention", 0) * (
#        output_dict["atom attention"]
#        / (output_dict.get("structure attention", 0) + output_dict["atom attention"])
#    )
#    output_dict["structure count distributed"] = output_dict.get(
#        "structure count", 0
#    ) + output_dict.get("bond count", 0) * (
#        output_dict.get("structure count", 0)
#        / (output_dict.get("structure count", 0) + output_dict.get("atom count", 0))
#    )
#    output_dict["atom count distributed"] = output_dict.get(
#        "atom count", 0
#    ) + output_dict.get("bond count", 0) * (
#        output_dict.get("atom count", 0)
#        / (output_dict.get("structure count", 0) + output_dict.get("atom count", 0))
#    )
#    return output_dict
#
#
#def aggregate_SELFIE_attention(line: List[Tuple[float, str]]) -> dict:
#    """Aggregate the attention from a SELFIES line according to similar tokens
#
#    Args:
#        line (List[Tuple[float, str]]): Line consisting of attention score and token
#
#    Returns:
#        dict: dict with keys to indicate aggregation and scores
#    """
#    output_dict = {}
#    structure_tokens = 0
#    high_attention = find_attention_outliers(pd.Series([entry[0] for entry in line]))
#    remaining_structure_tokens = 0
#    for counter, (score, token) in enumerate(line):
#        if structure_tokens > 0:
#            noting_token = str(token) + " overloaded"
#        else:
#            noting_token = token
#        if high_attention[counter]:
#            output_dict[f"high count {noting_token}"] = (
#                output_dict.get(f"high count {noting_token}", 0) + 1
#            )
#        output_dict[f"token count {noting_token}"] = (
#            output_dict.get(f"token count {noting_token}", 0) + 1
#        )
#        if remaining_structure_tokens > 0:
#            # overloaded tokens
#            remaining_structure_tokens -= 1
#            structure_score += score
#        elif "Ring" in token or "Branch" in token:
#            # parse overloading
#            output_dict["structure count"] = output_dict.get("structure count", 0) + 1
#            structure_tokens = int(token[-2])
#            remaining_structure_tokens = structure_tokens + 1
#            structure_score = score
#        else:
#            output_dict["atom attention"] = output_dict.get("atom attention", 0) + score
#            output_dict["atom count"] = output_dict.get("atom count", 0) + 1
#        if remaining_structure_tokens == 0 and structure_tokens > 0:
#            output_dict["structure attention"] = (
#                output_dict.get("structure attention", 0)
#                + structure_score / structure_tokens
#            )
#            structure_tokens = 0
#            structure_score = 0
#            # if "C]" in token:
#            #    if "C" == token[1]:
#            #        token = list(token)
#            #        token[1] = ""
#            #    output_dict[f"{token[1]}C count"] = (
#            #        output_dict.get(f"{token[1]}C count", 0) + 1
#            #    )
#            #    output_dict[f"{token[1]}C attention"] = (
#            #        output_dict.get(f"{token[1]}C attention", 0) + score
#            #    )
#    return output_dict
#
#
#def parse_att_dict(
#    SMILE_dict: dict, SELFIE_dict: dict, len_output: int, save_path: Path
#):
#    """Parse the attention dicts.
#
#    Args:
#        SMILE_dict (dict): dict of SMILES
#        SELFIE_dict (dict): dict of SELFIES
#        len_output (int): amount of samples
#        save_path (Path): path to save aggregates to
#    """
#    text = ""
#    for (representation, dikt) in [("SMILES", SMILE_dict), ("SELFIES", SELFIE_dict)]:
#        text += log_and_add(text, f"{representation}:")
#        for key in sorted(dikt.keys()):
#            if ("token" in key) or ("high" in key):
#                continue
#            text = log_and_add(text, f"The amount of {key} is {dikt[key]:.3f}.")
#            if "attention" in key:
#                modified_key = key.replace("attention", "count")
#                text = log_and_add(
#                    text,
#                    f"The amount of {key} per token is {dikt[key]/dikt[modified_key]:.3f}",
#                )
#            if "high" in key:
#                token = key[len("high count ") :]
#                token_count_key = "token count " + token
#                text = log_and_add(
#                    text,
#                    f"The percentage of high attention token: {token} is {dikt[key]/dikt[token_count_key]:.3f}",
#                )
#                text = log_and_add(
#                    text, f"The amount of {token} is {dikt[token_count_key]}"
#                )
#                continue
#            text = log_and_add(
#                text, f"The amount of {key} per sample is {dikt[key]/len_output:.3f}"
#            )
#    text = log_and_add(text, str(SMILE_dict))
#    text = log_and_add(text, str(SELFIE_dict))
#    with open(save_path, "w") as openfile:
#        openfile.write(text)
#
#
def load_molnet_test_set(task: str) -> Tuple[List[str], List[int]]:
    """Load MoleculeNet task

    Args:
        task (str): MoleculeNet task to load

    Returns:
        Tuple[List[str], List[int]]: Features, Labels
    """
    task_test = MOLNET_DIRECTORY[task]["load_fn"](
        featurizer=RawFeaturizer(smiles=True), splitter=MOLNET_DIRECTORY[task]["split"]
    )[1][2]
    task_SMILES = task_test.X
    task_labels = task_test.y
    return task_SMILES, task_labels


#def generate_attention_dict(
#    task: str, cuda: int
#) -> Tuple[List[List[float]], np.ndarray]:
#    """Generate the attention dict of a task
#
#    Args:
#        task (str): Task to find attention of
#        cuda (int): CUDA device to use
#
#    Returns:
#        Tuple[List[List[float]], np.ndarray]: attention, labels
#    """
#    task_SMILES, task_labels = load_molnet_test_set(task)
#    for encoding in [
#        "smiles_atom",
#        "selfies_atom",
#        "smiles_sentencepiece",
#        "selfies_sentencepiece",
#    ]:
#        specific_model_path = Path(
#            "/data/jgut/SoS_models/bbbp/smiles_atom_standard/5e-06_0.2_based_norm/checkpoint_best.pt"
#        )
#        data_path = TASK_PATH / task / encoding
#        data_path = Path("/data/jgut/SMILES_or_SELFIES/task/bbbp/smiles_atom_standard")
#        print(specific_model_path)
#        print(data_path)
#        
#        model = load_model(specific_model_path, data_path, cuda)
#        model.zero_grad()
#        data_path = data_path / "input0" / "test"
#        dataset = load_dataset(data_path)
#        source_dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))
#
#        assert len(task_SMILES) == len(
#            dataset
#        ), f"Real and filtered dataset {task} do not have same length."
#
#        text = [canonize_smile(smile) for smile in task_SMILES]
#        if encoding.startswith("selfies"):
#            text = [translate_selfie(smile)[0] for smile in text]
#
#        if encoding.endswith("sentencepiece"):
#            tokenizer = get_tokenizer(TOKENIZER_PATH / encoding)
#        else:
#            tokenizer = None
#        attention_encodings.append(
#            compute_model_output(
#                dataset,
#                model,
#                text,
#                source_dictionary,
#                False,
#                True,
#                False,
#                False,
#                tokenizer,
#            )[1]
#        )
#        print(attention_encodings)
#    output = list(zip(*attention_encodings))
#    labels = np.array(task_labels).transpose()[0]
#    return output, labels
#
#
#def aggregate_attention(output: List[Tuple[float, str]]) -> Tuple[dict, dict]:
#    """Aggregate the attention of annotated
#
#    Args:
#        output (str): List of attention and token
#
#    Returns:
#        Tuple[dict,dict]: SMILES_dict, SELFIES_dict
#    """
#    SMILE_dict = {}
#    SELFIE_dict = {}
#    for line in output:
#        curr_dict = aggregate_SMILE_attention(line[0])
#        SMILE_dict = {
#            key: SMILE_dict.get(key, 0) + curr_dict.get(key, 0)
#            for key in SMILE_dict | curr_dict
#        }
#        curr_dict = aggregate_SELFIE_attention(line[1])
#        SELFIE_dict = {
#            key: SELFIE_dict.get(key, 0) + curr_dict.get(key, 0)
#            for key in SELFIE_dict | curr_dict
#        }
#    return SMILE_dict, SELFIE_dict


#def produce_att_samples(
#    output: List[Tuple[float, str]],
#    labels: List[int],
#    save_dir: Path,
#    samples: int = 20,
#):
#    """Produce *samples* many samples from output with labels and save them to save_dir
#
#    Args:
#        output (List[Tuple[float, str]]): List of attention and token
#        labels (List[int]): Labels of samples
#        save_dir (Path): where to save samples to
#        samples (int, optional): amount of samples to select. Defaults to 20.
#    """
#    md = ""
#    for i in range(samples):
#        md += f"# Sample {i+1} with value {labels[i]:.3f}" + r"\n"
#        md += "## Molecule" + r"\n"
#        input_data = np.array([letter[1] for letter in output[i][0]])
#        md += f'{"".join(input_data)}' + r"\n"
#        md += (
#            f'{[token for token in re.split(PARSING_REGEX,create_identities("".join(input_data))[0]) if token] }'
#            + r"\n"
#        )
#        for it, tokenizer in enumerate(
#            ["SMILES", "SELFIES", "SMILES SentencePiece", "SMILES SentencePiece"]
#        ):
#            md += f"## {tokenizer}" + r"\n"
#            outliers = find_attention_outliers(
#                pd.Series([token[0] for token in output[i][it]])
#            )
#            input_data_corrected = np.array(
#                [
#                    (
#                        "*" + str(letter[1]) + "*"
#                        if outliers[counter]
#                        else str(letter[1]),
#                        f"{letter[0]-1/len(output[i][it]):.3f}",
#                    )
#                    for counter, letter in enumerate(output[i][it])
#                ]
#            )
#            input_data_pure = np.array(
#                [
#                    (
#                        "*" + str(letter[1]) + "*"
#                        if outliers[counter]
#                        else str(letter[1]),
#                        f"{letter[0]:.3f}",
#                    )
#                    for counter, letter in enumerate(output[i][it])
#                ]
#            )
#            df = pd.concat(
#                [
#                    pd.DataFrame(
#                        data=input_data_corrected[:, 1],
#                        index=input_data_corrected[:, 0],
#                    )
#                    .transpose()
#                    .reset_index(drop=True),
#                    pd.DataFrame(
#                        data=input_data_pure[:, 1], index=input_data_pure[:, 0]
#                    )
#                    .transpose()
#                    .reset_index(drop=True),
#                ]
#            )
#            md += df.to_markdown() + r"\n"
#    with open(save_dir, "w") as openfile:
#        openfile.write(md)

def to_markdown(molecule, smiles_atom, smiles_sentencepiece, selfies_atom, selfies_sentencepiece):
    md = ""
    md+=f'## Molecule'+"""\ """
    input_data = np.array([(letter[1], f"{letter[0]-1/len(molecule):.3f}") for letter in molecule])
    md+=f'{"".join(input_data[:,0])}'+"""\ """
    md+=f'## SMILES atomwise'+"""\ """
    input_data = np.array([(letter[1], f"{letter[0]-1/len(smiles_atom):.3f}") for letter in smiles_atom])
    input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in smiles_atom])
    df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()])
    md += df.to_markdown()+"""\ """
    md+=f'## SELFIES atomwise'+"""\ """
    input_data = np.array([(letter[1], f"{letter[0]-1/len(selfies_atom):.3f}") for letter in selfies_atom])
    input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in selfies_atom])
    df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()])
    md += df.to_markdown()+"""\ """
    md+=f'## SMILES SentencePiece'+"""\ """
    input_data = np.array([(letter[1], f"{letter[0]-1/len(smiles_sentencepiece):.3f}") for letter in smiles_sentencepiece])
    input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in smiles_sentencepiece])
    df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()]) 
    md += df.to_markdown()+"""\ """
    md+=f'## SELFIES SentencePiece'+"""\ """
    input_data = np.array([(letter[1], f"{letter[0]-1/len(selfies_sentencepiece):.3f}") for letter in selfies_sentencepiece])
    input_data2 = np.array([(letter[1], f"{letter[0]:.3f}") for letter in selfies_sentencepiece])
    df = pd.concat([pd.DataFrame(data = input_data[:,1], index = input_data[:,0]).transpose(),pd.DataFrame(data = input_data2[:,1], index = input_data2[:,0]).transpose()]) 
    md += df.to_markdown()+"""\ """
    return md

def gather_attention_model(input_mols, tokenizer_suffix):
    model_suffix = tokenizer_suffix+"_bart"
    fairseq_dict_path = TASK_PATH / "bbbp" /tokenizer_suffix
    tokenizer = get_tokenizer(TOKENIZER_PATH /tokenizer_suffix)
    model = load_model(PREDICTION_MODEL_PATH/model_suffix/"checkpoint_last.pt", fairseq_dict_path,None)
    source_dictionary = get_dictionary(FAIRSEQ_PREPROCESS_PATH/tokenizer_suffix/"dict.txt")
    #preprocessed = model.encode(input_mols)
    attended = compute_attention_output(model, [input_mols], source_dictionary, tokenizer)
    return attended

def gather_attention(SMILES):
    SMILES = canonize_smile(SMILES)
    SELFIES = translate_selfie(SMILES)[0]
    smiles_atom = gather_attention_model(SMILES, "smiles_atom_isomers")
    smiles_sentencepiece = gather_attention_model(SMILES, "smiles_trained_isomers")
    selfies_atom = gather_attention_model(SELFIES, "selfies_atom_isomers")
    selfies_sentencepiece = gather_attention_model(SELFIES, "selfies_trained_isomers")
    mkd = [(1/len(SMILES), letter) for letter in SMILES]
    markdown = to_markdown(mkd,smiles_atom[0], smiles_sentencepiece[0], selfies_atom[0], selfies_sentencepiece[0])
    return markdown

#if __name__ == "__main__":
#    attention_encodings = []
#    args = parse_arguments(True, False, True)
#    assert args["task"] in list(
#        MOLNET_DIRECTORY.keys()
#    ), f"{args['task']} not in MOLNET tasks."
#    output, labels = generate_attention_dict(args["task"], args["cuda"])
#    SMILE_dict, SELFIE_dict = aggregate_attention(output)
#    parse_att_dict(
#        SMILE_dict, SELFIE_dict, len(output), f"logs/attention_agg_{args['task']}.txt"
#    )
#    produce_att_samples(output, labels, f"logs/attention_samples_{args['task']}.md")
