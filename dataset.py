""" Dataset class for loading
SMILES or SELFIES, 2022
"""
import logging
import os
import re
from pathlib import Path

import pandas as pd
from deepchem.feat import RawFeaturizer
from tqdm import tqdm

from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    MOLNET_DIRECTORY,
    TASK_PATH,
    TOKENIZER_PATH,
    USPTO_PATH,
)
from tokenisation import get_tokenizer, tokenize_dataset

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
        read_data = pd.read_csv(
            USPTO_PATH / "lef_uspto" / ("filtered_" + dataset + ".txt"),
            delimiter=" ",
            names=["reaction", "delimiters"],
        )["reaction"]
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
            f'fairseq-preprocess --trainpref {output_dir/"train"} --validpref {output_dir/"valid"} --testpref {output_dir/"test"} --srcdict {model_dict} --tgtdict {model_dict} --workers 60 --source-lang input --target-lang label --destdir {output_dir/"pre-processed"}'
        )
    )


def prepare_molnet(
    task: str,
    tokenizer,
    selfies: bool,
    output_dir: Path,
    model_dict: Path,
):
    """Prepare Molnet tasks with fairseq, so that they can be used for fine-tuning.

    Args:
        task (str): which MolNet task to prepare
        tokenizer (tokenizer): which tokenizer to use for this dataset
        selfies (bool): Use selfies or not; should agree with selected tokenizer
        output_dir (Path): where to save preprocessed files
        model_dict (Path): which vocabulary to use for pre-processing
    """
    molnet_infos = MOLNET_DIRECTORY[task]
    _, splits, _ = molnet_infos["load_fn"](
        featurizer=RawFeaturizer(smiles=True), splitter=molnet_infos["split"]
    )
    tasks = ["train", "valid", "test"]
    for id_number, split in enumerate(splits):
        mol = tokenize_dataset(tokenizer, split.X, selfies)
        # no normalisation of labels
        if "tasks_wanted" in molnet_infos:
            correct_column = split.tasks.tolist().index(molnet_infos["tasks_wanted"][0])
            label = split.y[:, correct_column]
        else:
            label = split.y
        label = label[~pd.isna(mol)]
        logging.info(
            f"For task {task} in set {tasks[id_number]}, {sum(pd.isna(mol))} ({(sum(pd.isna(mol))/len(mol))*100:.2f})% samples could not be formed to SELFIES."
        )
        mol = mol[~pd.isna(mol)]
        mol.tofile(output_dir / (tasks[id_number] + ".input"), sep="\n", format="%s")
        label.tofile(output_dir / (tasks[id_number] + ".label"), sep="\n", format="%s")
        if molnet_infos["dataset_type"] == "regression":
            (output_dir / "label").mkdir(parents=True, exist_ok=True)
            label.tofile(
                output_dir / "label" / (tasks[id_number] + ".label"),
                sep="\n",
                format="%s",
            )
    os.system(
        (
            f'fairseq-preprocess --only-source --trainpref {output_dir/"train.input"} --validpref {output_dir/"valid.input"} --testpref {output_dir/"test.input"} --destdir {output_dir/"input0"} --srcdict {model_dict} --workers 60'
        )
    )
    os.system(
        (
            f'fairseq-preprocess --only-source --trainpref {output_dir/"train.label"} --validpref {output_dir/"valid.label"} --testpref {output_dir/"test.label"} --destdir {output_dir/"label"} --workers 60'
        )
    )


if __name__ == "__main__":
    molnets = MOLNET_DIRECTORY
    for tokenizer_suffix in [
        "smiles_isomers_atom",
        "smiles_isomers_sentencepiece",
        "selfies_isomers_atom",
        "selfies_isomers_sentencepiece",
    ]:
        selfies = tokenizer_suffix.startswith("selfies")
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        preprocess_path = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
        for key in MOLNET_DIRECTORY:
            output_dir = TASK_PATH / key / tokenizer_suffix
            output_dir.mkdir(parents=True, exist_ok=True)
            prepare_molnet(key, tokenizer, selfies, output_dir, preprocess_path)
            logging.info(f"Finished creating {output_dir}")

        for key in ["jin", "schwaller", "lef"]:
            output_dir = TASK_PATH / key / tokenizer_suffix
            output_dir.mkdir(parents=True, exist_ok=True)
            prepare_uspto(key, tokenizer, selfies, output_dir, preprocess_path)
            logging.info(f"Finished creating {output_dir}")
