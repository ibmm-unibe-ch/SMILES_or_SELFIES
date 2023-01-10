""" Dataset class for loading
SMILES or SELFIES, 2022
"""
import logging
import os
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch
from deepchem.feat import RawFeaturizer
from torch.utils.data import Dataset, random_split

from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    MOLNET_DIRECTORY,
    SEED,
    TASK_PATH,
    TOKENIZER_PATH,
)
from tokenisation import get_tokenizer, tokenize_dataset


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


def split_train_eval(
    dataset: Dataset, eval_size: Union[int, float] = 10000
) -> Tuple[Dataset, Dataset]:
    """Split torch dataset to eval_size

    Args:
        dataset (Dataset): dataset to split
        eval_size (Union[int, float], optional): relative or absolute size of eval_set. Defaults to 10000.

    Returns:
        Tuple[Dataset, Dataset]: train_set, eval_set
    """
    len_dataset = 1 if eval_size < 0 else len(dataset)
    train_set, eval_set = random_split(
        dataset,
        [len_dataset - eval_size, eval_size],
        generator=torch.Generator().manual_seed(SEED + 238947),
    )
    return train_set, eval_set


def prepare_molnet(
    task: str, tokenizer, selfies: bool, output_dir: Path, model_dict: Path
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
    tokenizer_suffixes = [
        "selfies_sentencepiece",
        "smiles_sentencepiece",
        "smiles_atom",
        "selfies_atom",
    ]
    molnets = MOLNET_DIRECTORY
    del molnets["pcba"]
    for tokenizer_suffix in tokenizer_suffixes:
        selfies = tokenizer_suffix.startswith("selfies")
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        output_dir = TASK_PATH / tokenizer_suffix
        for key in MOLNET_DIRECTORY:
            (output_dir / key).mkdir(parents=True, exist_ok=True)
            prepare_molnet(
                key,
                tokenizer,
                selfies,
                output_dir / key,
                FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt",
            )
            logging.info(f"Finished creating {output_dir}")
