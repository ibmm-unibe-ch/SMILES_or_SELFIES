""" Tokenisation
SMILES or SELFIES, 2022
"""

import logging
from pathlib import Path

import pandas as pd
from tokenizers import (
    Regex,
    SentencePieceUnigramTokenizer,
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)
from transformers import BartTokenizerFast

from constants import TOKENIZER_PATH


def train_sentencepiece(
    training_data: pd.Series, save_path: Path, vocab_size: int = 1000
) -> BartTokenizerFast:
    """Train a sentencepiece classifier on training_data and save to save_path

    Args:
        training_data (pd.Series): data to train on
        save_path (Path): path to save tokenizer to
        vocab_size (int, optional): size of vocab. Defaults to 1000.

    Returns:
        BartTokenizerFast: Wrapped tokenizer
    """
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
    # BERTpost-processor needed?
    tokenizer.save_pretrained(save_path)
    return tokenizer


def train_atomwise_tokenizer(
    training_data: pd.Series, save_path: Path, vocab_size: int = 1000
) -> BartTokenizerFast:
    """Train "normal"/atomwise tokenizer on training_data and save to save_path

    Args:
        training_data (pd.Series): data to train on
        save_path (Path): path to save to
        vocab_size (int, optional): size of vocabulary. Defaults to 1000.

    Returns:
        BartTokenizerFast: Wrapped tokenizer
    """
    tk_tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    # copied from https://colab.research.google.com/drive/1tsiTpC4i26QNdRzBHFfXIOFVToE54-9b?usp=sharing#scrollTo=UHzrWuFpCtzs
    # same in DeepChem
    splitting_regex = Regex(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    )
    tk_tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=splitting_regex, behavior="isolated"
    )
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, show_progress=True, special_tokens=special_tokens
    )
    tk_tokenizer.train_from_iterator(training_data, trainer=trainer)
    tokenizer = BartTokenizerFast(tokenizer_object=tk_tokenizer)
    logging.info(f"Saving tokenizer to {save_path}")
    tokenizer.save_pretrained(save_path)
    return tokenizer


def get_tokenizer(tokenizer_path: Path) -> BartTokenizerFast:
    """load tokenizer from tokenizer_path, works for all tokenizers

    Args:
        tokenizer_path (Path): path to load from

    Returns:
        BartTokenizerFast: loaded tokenizer
    """
    return BartTokenizerFast.from_pretrained(tokenizer_path)


if __name__ == "__main__":
    # SMILES = pd.read_csv("processed/10m_dataframe.csv", usecols=[212]).values
    # atom_SMILES_tokenizer = train_atomwise_tokenizer(
    #    SMILES, TOKENIZER_PATH / "Atom_SMILES", vocab_size=1000
    # )
    # print(
    #    atom_SMILES_tokenizer(
    #        "c1ccc(-c2cccc3c2c2c4oc5c(ccc6c5c5ccccc5n6-c5ccccc5)c4ccc2n3-c2ccccc2)"
    #    )
    # )
    # SELFIES = pd.read_csv("processed/10m_dataframe.csv", usecols=[210]).values
    # atom_SELFIES_tokenizer = train_atomwise_tokenizer(
    #    SELFIES, TOKENIZER_PATH / "atom_SELFIES", vocab_size=1000
    # )
    # print(
    #    atom_SELFIES_tokenizer(
    #        "[C][C][C][C][C][Branch1][=C][N][C][=C][C][=C][Branch1][C][C][C][=C][Ring1][#Branch1][Cl][C][=Branch1][C][=O][O][C]"
    #    )
    # )
    SMILES = pd.read_csv("processed/10m_dataframe.csv", usecols=[212]).values
    SMILES_tokenizer = train_sentencepiece(
        SMILES, TOKENIZER_PATH / "SMILES", vocab_size=1000
    )
    print(
        SMILES_tokenizer(
            "c1ccc(-c2cccc3c2c2c4oc5c(ccc6c5c5ccccc5n6-c5ccccc5)c4ccc2n3-c2ccccc2)"
        )
    )
    SELFIES = pd.read_csv("processed/10m_dataframe.csv", usecols=[210]).values
    SELFIES_tokenizer = train_sentencepiece(
        SELFIES, TOKENIZER_PATH / "SELFIES", vocab_size=1000
    )
    print(
        SELFIES_tokenizer(
            "[C][C][C][C][C][Branch1][=C][N][C][=C][C][=C][Branch1][C][C][C][=C][Ring1][#Branch1][Cl][C][=Branch1][C][=O][O][C]"
        )
    )
    SELFIES = pd.read_csv("processed/10m_dataframe.csv", usecols=[210]).values
    atom_SELFIES_tokenizer = train_atomwise_tokenizer(
        SELFIES, TOKENIZER_PATH / "atom_SELFIES", vocab_size=1000
    )
    print(
        atom_SELFIES_tokenizer(
            "[C][C][C][C][C][Branch1][=C][N][C][=C][C][=C][Branch1][C][C][C][=C][Ring1][#Branch1][Cl][C][=Branch1][C][=O][O][C]"
        )
    )
