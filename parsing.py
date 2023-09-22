""" Parsing of files with tokenisation
SMILES or SELFIES, 2022
"""

import pandas as pd
from constants import PROCESSED_PATH, SEED, TOKENIZER_PATH, VAL_SIZE
from sklearn.utils import shuffle
from tokenisation import get_tokenizer
from tqdm import tqdm

if __name__ == "__main__":
    for subset in ["isomers", "standard"]:
        for encoding in ["SELFIES","SMILES", "OWN"]:
            column = pd.read_csv(
                PROCESSED_PATH / subset / f"full_deduplicated_{subset}.csv",
                usecols=[encoding],
            ).values.tolist()
            encoding = encoding.lower()
            column = shuffle(column, random_state=SEED - 385)
            trained_tokenizer = get_tokenizer(
                TOKENIZER_PATH / f"{encoding}_trained_{subset}"
            )
            atom_tokenizer = get_tokenizer(TOKENIZER_PATH / f"{encoding}_atom_{subset}")
            val_size = VAL_SIZE
            for value in tqdm(column):
                val_str = value[0]
                atom_tokens = atom_tokenizer.convert_ids_to_tokens(
                    atom_tokenizer(val_str).input_ids
                )
                trained_tokens = trained_tokenizer.convert_ids_to_tokens(
                    trained_tokenizer(val_str).input_ids
                )
                if val_size > 0:
                    with open(
                        PROCESSED_PATH / f"{encoding}_atom_{subset}_val", "a"
                    ) as open_file:
                        open_file.write(" ".join(atom_tokens) + "\n")
                    with open(
                        PROCESSED_PATH / f"{encoding}_trained_{subset}_val", "a"
                    ) as open_file:
                        open_file.write(" ".join(trained_tokens) + "\n")
                else:
                    with open(
                        PROCESSED_PATH / f"{encoding}_atom_{subset}", "a"
                    ) as open_file:
                        open_file.write(" ".join(atom_tokens) + "\n")
                    with open(
                        PROCESSED_PATH / f"{encoding}_trained_{subset}", "a"
                    ) as open_file:
                        open_file.write(" ".join(trained_tokens) + "\n")
                val_size -= 1
