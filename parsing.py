""" Parsing of files with tokenisation
SMILES or SELFIES, 2022
"""

import pandas as pd
from constants import PROCESSED_PATH, SEED, TOKENIZER_PATH, VAL_SIZE
from sklearn.utils import shuffle
from tokenisation import get_tokenizer
from tqdm import tqdm

if __name__ == "__main__":
    smiles = pd.read_csv(
        PROCESSED_PATH/"own" / "10m_deduplicated_isomers.csv",
        usecols=[str(1)],
    ).values.tolist()
    smiles = shuffle(smiles, random_state=SEED - 385)
    smiles_trained_tokenizer = get_tokenizer(TOKENIZER_PATH / "smiles_sentencepiece")
    smiles_atom_tokenizer = get_tokenizer(TOKENIZER_PATH / "smiles_atom")
    val_size = VAL_SIZE
    for value in tqdm(smiles):
        val_str = value[0]
        atom_tokens = smiles_atom_tokenizer.convert_ids_to_tokens(
            smiles_atom_tokenizer(val_str).input_ids
        )
        trained_tokens = smiles_trained_tokenizer.convert_ids_to_tokens(
            smiles_trained_tokenizer(val_str).input_ids
        )
        if val_size > 0:
            # with open(PROCESSED_PATH / "smiles_isomers_val", "a") as open_file:
            #    open_file.write(" ".join(atom_tokens) + "\n")
            with open(PROCESSED_PATH / "trained_smiles_isomers_val", "a") as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")

        else:
            # with open(PROCESSED_PATH / "smiles_isomers_train", "a") as open_file:
            #    open_file.write(" ".join(atom_tokens) + "\n")
            with open(
                PROCESSED_PATH / "trained_smiles_isomers_train", "a"
            ) as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")
        val_size -= 1

    selfies = pd.read_csv(
        PROCESSED_PATH / "10m_deduplicated_isomers.csv",
        usecols=[str(208)],
    ).values.tolist()
    selfies = shuffle(selfies, random_state=SEED - 385)
    selfies_trained_tokenizer = get_tokenizer(
        TOKENIZER_PATH / "selfies_sentencepiece_isomers"
    )

    selfies_atom_tokenizer = get_tokenizer(TOKENIZER_PATH / "selfies_atom_isomers")
    val_size = VAL_SIZE
    for value in tqdm(selfies):
        val_str = value[0]
        atom_tokens = selfies_atom_tokenizer.convert_ids_to_tokens(
            selfies_atom_tokenizer(val_str).input_ids
        )
        trained_tokens = selfies_trained_tokenizer.convert_ids_to_tokens(
            selfies_trained_tokenizer(val_str).input_ids
        )
        if val_size > 0:
            # with open(PROCESSED_PATH / "selfies_isomers_val", "a") as open_file:
            #    open_file.write(" ".join(atom_tokens) + "\n")
            with open(PROCESSED_PATH / "trained_selfies_isomers_val", "a") as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")
        else:
            # with open(PROCESSED_PATH / "selfies_isomers_train", "a") as open_file:
            #    open_file.write(" ".join(atom_tokens) + "\n")
            with open(
                PROCESSED_PATH / "trained_selfies_isomers_train",
                "a",
            ) as open_file:
                open_file.write(" ".join(trained_tokens) + "\n")
        val_size -= 1
