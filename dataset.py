""" Dataset class for loading
SMILES or SELFIES, 2022
"""
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from deepchem.feat import RawFeaturizer

from torch.utils.data import Dataset


from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    MOLNET_DIRECTORY,
    TASK_PATH,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
)
from tokenisation import get_tokenizer, tokenize_dataset

os.environ["MKL_THREADING_LAYER"] = "GNU"


class PandasDataset(Dataset):
    """Simple wrapper to load pandas to a torch dataset"""

    def __init__(self, file_path: Path, column: int, tokenizer):
        self.df = pd.read_csv(file_path, usecols=[str(column)]).values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.tokenizer(self.df[idx][0], padding="max_length", max_length=42)
        item["labels"] = item["input_ids"].copy()
        for key in item:
            item[key] = torch.tensor(item[key])
        return item


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

def prepare_molnet(
    task: str,
    tokenizer,
    selfies: bool,
    output_dir: Path,
    model_dict: Path,
):
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
    for tokenizer_suffix in TOKENIZER_SUFFIXES:
        selfies = tokenizer_suffix.startswith("selfies")
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        for key in MOLNET_DIRECTORY:
            output_dir = TASK_PATH / key / (tokenizer_suffix)
            output_dir.mkdir(parents=True, exist_ok=True)
            prepare_molnet(
                key,
                tokenizer,
                selfies,
                output_dir,
                FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt",
            )
            logging.info(f"Finished creating {output_dir}")
