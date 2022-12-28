import pandas as pd
from sklearn.utils import shuffle
from SmilesPE.pretokenizer import atomwise_tokenizer
from tqdm import tqdm

from constants import SEED, VAL_SIZE

if __name__ == "__main__":
    smiles = pd.read_csv(
        "/home/jgut/GitHub/SMILES_or_SELFIES/processed/10m_deduplicated.csv",
        usecols=[str(210)],
    ).values.tolist()
    smiles = shuffle(smiles, random_state=SEED - 385)
    val_size = VAL_SIZE
    for value in tqdm(smiles):
        val_str = value[0]
        atom_tokens = atomwise_tokenizer(val_str)
        # sentencepiece_tokens = sentencepiece_tokenizer(val_str)
        if val_size > 0:
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/smiles_val", "a"
            ) as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            """
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/sentencepiece_smiles_val", "a"
            ) as open_file:
                open_file.write(" ".join(sentencepiece_tokens) + "\n")
            """

        else:
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/smiles_train", "a"
            ) as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            """
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/sentencepiece_smiles_train", "a"
            ) as open_file:
                open_file.write(" ".join(sentencepiece_tokens) + "\n")
            """
        val_size -= 1

    selfies = pd.read_csv(
        "/home/jgut/GitHub/SMILES_or_SELFIES/processed/10m_deduplicated.csv",
        usecols=[str(208)],
    ).values.tolist()
    selfies = shuffle(selfies, random_state=SEED - 385)
    val_size = VAL_SIZE
    for value in tqdm(selfies):
        val_str = value[0]
        atom_tokens = atomwise_tokenizer(val_str)
        # sentencepiece_tokens = sentencepiece_tokenizer(val_str)
        if val_size > 0:
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/selfies_val", "a"
            ) as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            """
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/sentencepiece_selfies_val", "a"
            ) as open_file:
                open_file.write(" ".join(sentencepiece_tokens) + "\n")
            """

        else:
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/selfies_train", "a"
            ) as open_file:
                open_file.write(" ".join(atom_tokens) + "\n")
            """
            with open(
                "/home/jgut/GitHub/SMILES_or_SELFIES/processed/sentencepiece_selfies_train", "a"
            ) as open_file:
                open_file.write(" ".join(sentencepiece_tokens) + "\n")
            """
        val_size -= 1
