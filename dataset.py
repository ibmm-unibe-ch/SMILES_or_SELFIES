""" Dataset class for loading
SMILES or SELFIES, 2022
"""
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset, random_split
from constants import SEED
from typing import Union
import torch


class PandasDataset(Dataset):
    """Simple wrapper to load pandas to a torch dataset"""

    def __init__(self, file_path: Path, column: int):
        self.df = pd.read_csv(file_path, usecols=[column]).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx]


def split_train_eval(dataset: Dataset, eval_size: Union[int, float] = 10000) -> Dataset:
    len_dataset = 1 if eval_size < 0 else len(dataset)
    train_set, eval_set = random_split(
        dataset,
        [len_dataset - eval_size, eval_size],
        generator=torch.Generator().manual_seed(SEED + 238947),
    )
    return train_set, eval_set


if __name__ == "__main__":
    """only for testing purposes"""
    test = PandasDataset(
        "/home/jannik-gut/GitHub/SMILES_or_SELFIES/processed/10m_dataframe.csv", 210
    )
    train, test = split_train_eval(test, 10)
    print(len(train))
    print(len(test))
    print(test[0])
