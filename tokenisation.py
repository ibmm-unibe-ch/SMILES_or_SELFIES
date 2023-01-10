""" Tokenisation
SMILES or SELFIES, 2022
"""

import json
import logging
from pathlib import Path

import numpy as np
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
from preprocessing import translate_selfie


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
    tok = BartTokenizerFast.from_pretrained(tokenizer_path)
    return tok


def tokenize_with_space(tokenizer, sample_smiles: str, selfies: False) -> str:
    translated_selfie, length = translate_selfie(str(sample_smiles))
    if translated_selfie is None:
        return None
    if selfies:
        sample_smiles = translated_selfie
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(str(sample_smiles)).input_ids)
    return " ".join(tokens)


def tokenize_dataset(tokenizer, dataset: pd.Series, selfies=False) -> pd.Series:
    output = np.array(
        [tokenize_with_space(tokenizer, sample, selfies) for sample in dataset]
    )
    return output


def create_dict_from_fairseq(fairseq_dict_dir: Path, output_path: Path):
    # hopefully never needed
    output = dict_stub
    dict_stub = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    # as seen in https://huggingface.co/facebook/bart-base/
    # <mask> needed?
    before = len(dict_stub)
    with open(fairseq_dict_dir, "r") as open_file:
        entries = open_file.readlines()
    for entry_number, entry_string in enumerate(entries):
        output[entry_string.split(" ")[0].strip()] = before + entry_number
    with open(output_path, "w") as outfile:
        json.dump(output, outfile)
    return output


# tokenizer = BartTokenizer("sample.json", "merges.txt" --> can be empty)

if __name__ == "__main__":
    SMILES = pd.read_csv("processed/10m_dataframe.csv", usecols=[212]).values
    atom_SMILES_tokenizer = train_atomwise_tokenizer(
        SMILES, TOKENIZER_PATH / "smiles_atom", vocab_size=1000
    )
    SELFIES = pd.read_csv("processed/10m_dataframe.csv", usecols=[210]).values
    atom_SELFIES_tokenizer = train_atomwise_tokenizer(
        SELFIES, TOKENIZER_PATH / "selfies_atom", vocab_size=1000
    )
    """
    tk_tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok._tokenizer.post_processor = BertProcessing(
                ("</s>", 2),
                ("<s>", 0),
        )
    """
    SMILES = pd.read_csv("processed/10m_dataframe.csv", usecols=[212]).values
    SMILES_tokenizer = train_sentencepiece(
        SMILES, TOKENIZER_PATH / "smiles_sentencepiece", vocab_size=1000
    )
    SELFIES = pd.read_csv("processed/10m_dataframe.csv", usecols=[210]).values
    SELFIES_tokenizer = train_sentencepiece(
        SELFIES, TOKENIZER_PATH / "selfies_sentencepiece", vocab_size=1000
    )
