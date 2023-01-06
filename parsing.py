""" Parsing of files with tokenisation
SMILES or SELFIES, 2022
"""

import pandas as pd
from sklearn.utils import shuffle
from SmilesPE.pretokenizer import atomwise_tokenizer
from tqdm import tqdm

from constants import PROCESSED_PATH, SEED, TOKENIZER_PATH, VAL_SIZE
from tokenisation import get_tokenizer

if __name__ == "__main__":
    smiles = pd.read_csv(
        PROCESSED_PATH / "10m_deduplicated.csv",
        usecols=[str(210)],
    ).values.tolist()
    smiles = shuffle(smiles, random_state=SEED - 385)
    smiles_trained_tokenizer = get_tokenizer(TOKENIZER_PATH / "SMILES")
    # smiles_atom_tokenizer = get_tokenizer(TOKENIZER_PATH/atom_SMILES) instead of atomwise_tokenizer
    val_size = VAL_SIZE
    for value in tqdm(smiles):
        val_str = value[0]
        atom_tokens = atomwise_tokenizer(val_str)
        trained_tokens = smiles_trained_tokenizer.convert_ids_to_tokens(
            smiles_trained_tokenizer(val_str).input_ids
        )
        if val_size > 0:
            with open(PROCESSED_PATH / "smiles_val", "a") as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            with open(PROCESSED_PATH / "trained_smiles_val", "a") as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")

        else:
            with open(PROCESSED_PATH / "smiles_train", "a") as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            with open(PROCESSED_PATH / "trained_smiles_train", "a") as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")
        val_size -= 1

    selfies = pd.read_csv(
        PROCESSED_PATH / "10m_deduplicated.csv",
        usecols=[str(208)],
    ).values.tolist()
    selfies = shuffle(selfies, random_state=SEED - 385)
    val_size = VAL_SIZE
    selfies_trained_tokenizer = get_tokenizer(TOKENIZER_PATH / "SELFIES")

    for value in tqdm(selfies):
        val_str = value[0]
        atom_tokens = atomwise_tokenizer(val_str)
        trained_tokens = selfies_trained_tokenizer.convert_ids_to_tokens(
            selfies_trained_tokenizer(val_str).input_ids
        )

        if val_size > 0:
            with open(PROCESSED_PATH / "selfies_val", "a") as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            with open(PROCESSED_PATH / "trained_selfies_val", "a") as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")

        else:
            with open(PROCESSED_PATH / "selfies_train", "a") as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            with open(
                PROCESSED_PATH / "trained_selfies_train",
                "a",
            ) as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")
        val_size -= 1
