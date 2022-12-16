""" Tokenisation 
SMILES or SELFIES, 2022
"""

import logging
from pathlib import Path

import pandas as pd
from SmilesPE.pretokenizer import atomwise_tokenizer
from tokenizers import SentencePieceUnigramTokenizer
from transformers import BartTokenizerFast

from constants import TOKENIZER_PATH


def train_sentencepiece(
    training_data: pd.Series, save_path: Path, vocab_size: int = 1000
) -> BartTokenizerFast:
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
    tk_tokenizer = SentencePieceUnigramTokenizer()
    tk_tokenizer.train_from_iterator(
        training_data,
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=special_tokens,
        unk_token="<unk>",
        length=len(training_data),
    )
    tokenizer = BartTokenizerFast(tokenizer_object=tk_tokenizer)
    logging.info(f"Saving tokenizer to {save_path}")
    tokenizer.save_pretrained(save_path)
    return tokenizer


class Atomwise_Tokenizer(object):
    """Run atom-level SMILES tokenization"""

    def __init__(self):
        """Constructs a atom-level Tokenizer."""

    def tokenize(self, text):
        """Basic Tokenization of a SMILES."""
        return atomwise_tokenizer(text)


def get_atomwise_tokenizer() -> BartTokenizerFast:
    tk_tokenizer = Atomwise_Tokenizer()
    tokenizer = BartTokenizerFast(tokenizer_object=tk_tokenizer)
    return tokenizer


def get_sentencepiece_tokenizer(tokenizer_path: Path) -> BartTokenizerFast:
    return BartTokenizerFast.from_pretrained(tokenizer_path)


if __name__ == "__main__":
    SMILES = pd.read_csv("processed/10m_dataframe.csv", usecols=[212]).values
    SMILES_tokenizer = train_sentencepiece(
        SMILES, TOKENIZER_PATH / "SMILES", vocab_size=1000
    )
    # print(
    #    SMILES_tokenizer.tokenize(
    #        "c1ccc(-c2cccc3c2c2c4oc5c(ccc6c5c5ccccc5n6-c5ccccc5)c4ccc2n3-c2ccccc2)cc1"
    #    )
    # )
    SELFIES = pd.read_csv("processed/10m_dataframe.csv", usecols=[210]).values
    SELFIES_tokenizer = train_sentencepiece(
        SELFIES, TOKENIZER_PATH / "SELFIES", vocab_size=1000
    )
    # print(
    #    SELFIES_tokenizer.tokenize(
    #        "[C][C][C][C][C][Branch1][=C][N][C][=C][C][=C][Branch1][C][C][C][=C][Ring1][#Branch1][Cl][C][=Branch1][C][=O][O][C]"
    #    )
    # )
    # atom_tokenizer = get_atomwise_tokenizer()
    # print(
    #    atom_tokenizer.tokenize(
    #        "c1ccc(-c2cccc3c2c2c4oc5c(ccc6c5c5ccccc5n6-c5ccccc5)c4ccc2n3-c2ccccc2)cc1"
    #    )
    # )
    # print(
    #    atom_tokenizer.tokenize(
    #        "[C][C][C][C][C][Branch1][=C][N][C][=C][C][=C][Branch1][C][C][C][=C][Ring1][#Branch1][Cl][C][=Branch1][C][=O][O][C]"
    #    )
    # )
