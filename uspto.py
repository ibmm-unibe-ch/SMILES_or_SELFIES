""" Preparation of USPTO dataset for reaction prediction
SMILES or SELFIES, 2023
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    TASK_PATH,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
    USPTO_PATH,
)
from tokenisation import get_tokenizer, tokenize_dataset
from tqdm import tqdm

os.environ["MKL_THREADING_LAYER"] = "GNU"


def split_reaction(smirks):
    agents_check = re.findall(">>", smirks)
    if len(agents_check) == 0:
        split_smirks = smirks.split(">")
        reactant = split_smirks[0]
        agent = split_smirks[1]
        product = split_smirks[2]
    elif len(agents_check) == 1:
        split_smirks = smirks.split(">>")
        agent = ""
        reactant = split_smirks[0]
        product = split_smirks[1]
    else:
        return None
    return (reactant, agent, product)


def split_it(ds):
    reactants = []
    agents = []
    products = []
    smirks_orig = []
    for smirks in tqdm(ds):
        reactant, agent, product = split_reaction(str(smirks))
        reactants.append(reactant)
        agents.append(agent)
        products.append(product)
        smirks_orig.append(smirks)
    df = pd.DataFrame(
        {
            "SMIRKS": smirks_orig,
            "Reactant": reactants,
            "Agent": agents,
            "Product": products,
        }
    )
    return df


def read_uspto(task: str, dataset: str):
    if task == "jin":
        read_data = pd.read_csv(
            USPTO_PATH
            / "ReactionSeq2Seq_Dataset"
            / ("Jin_USPTO_1product_" + dataset + ".txt"),
            delimiter=" ",
            skiprows=1,
            names=["reaction", "delimiters"],
        )["reaction"]
    elif task == "schwaller":
        read_data = pd.read_csv(
            USPTO_PATH
            / "ReactionSeq2Seq_Dataset"
            / ("US_patents_1976-Sep2016_1product_reactions_" + dataset + ".csv"),
            skiprows=2,
            header=0,
            sep="\t",
            usecols=["CanonicalizedReaction"],
        )["CanonicalizedReaction"]
    elif task == "lef":
        with open(
            USPTO_PATH / "lef_uspto" / ("filtered_" + dataset + ".txt"), "r"
        ) as open_file:
            reactions = open_file.readlines()
        read_data = reactions
    else:
        raise Exception(f"Wrong USPTO task name: {task}")
    return split_it(read_data)


def prepare_uspto(
    task: str,
    tokenizer,
    selfies: bool,
    output_dir: Path,
    model_dict: Path,
):
    """ """
    datasets = ["train", "valid", "test"]
    for dataset in datasets:
        read_data = read_uspto(task, dataset)
        reactant = tokenize_dataset(tokenizer, read_data["Reactant"], selfies)
        products = tokenize_dataset(tokenizer, read_data["Product"], selfies)
        clean_reactant = reactant[~(pd.isna(reactant) | pd.isna(products))]
        clean_products = products[~(pd.isna(reactant) | pd.isna(products))]
        logging.info(
            f"For task {task} in set {dataset}, {sum(pd.isna(clean_products))} ({(sum(pd.isna(clean_products))/len(products))*100:.2f})% samples could not be formed to SELFIES."
        )
        clean_reactant.tofile(output_dir / (dataset + ".input"), sep="\n", format="%s")
        clean_products.tofile(output_dir / (dataset + ".label"), sep="\n", format="%s")
    os.system(
        (
            f'fairseq-preprocess --only-source --trainpref {output_dir/"train.input"} --validpref {output_dir/"valid.input"} --testpref {output_dir/"test.input"} --destdir {output_dir/"input0"} --srcdict {model_dict} --workers 60'
        )
    )
    os.system(
        (
            f'fairseq-preprocess --only-source --trainpref {output_dir/"train.label"} --validpref {output_dir/"valid.label"} --testpref {output_dir/"test.label"} --destdir {output_dir/"label"} --srcdict {model_dict} --workers 60'
        )
    )


if __name__ == "__main__":
    tasks = ["jin", "schwaller", "lef"]
    for task in tasks:
        for tokenizer_suffix in TOKENIZER_SUFFIXES:
            selfies = tokenizer_suffix.startswith("selfies")
            tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)

            output_dir = TASK_PATH / task / (tokenizer_suffix)
            output_dir.mkdir(parents=True, exist_ok=True)
            prepare_uspto(
                task,
                tokenizer,
                selfies,
                output_dir,
                FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt",
            )
