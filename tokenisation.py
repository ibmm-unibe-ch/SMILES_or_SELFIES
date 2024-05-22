""" Tokenisation
SMILES or SELFIES, 2022
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from constants import PROCESSED_PATH, TOKENIZER_PATH
from preprocessing import canonize_smile, translate_selfie
from rdkit import RDLogger
from tokenizers import (
    Regex,
    SentencePieceUnigramTokenizer,
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)
from tqdm import tqdm
from transformers import BartTokenizerFast

RDLogger.DisableLog("rdApp.warning")


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


def tokenize_with_space(tokenizer, sample_smiles: str, selfies=False) -> str:
    """Tokenize sample with tokenizer

    Args:
        tokenizer (Huggingface Tokenizer): Tokenizer to use for tokenisation
        sample_smiles (str): string to tokenize
        selfies (bool, optional): Tranlsate to SELFIES (True) or keep SMILES(False). Defaults to False.
       
    Returns:
        str: Tokenized sample
    """
    if (
        translate_selfie(str(sample_smiles))[0] is None
    ):
        return None
    canon_smiles = canonize_smile(sample_smiles)
    if selfies:
        canon_smiles = translate_selfie(str(canon_smiles))[0]
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(str(canon_smiles)).input_ids)
    return " ".join(tokens)


def tokenize_dataset(tokenizer, dataset: pd.Series, selfies=False) -> pd.Series:
    """Tokenize whole dataset with tokenizer

    Args:
        tokenizer (_type_): Tokenizer to use for tokenisation
        dataset (pd.Series): dataset to tokenize
        selfies (bool, optional): Tranlsate to SELFIES (True) or keep SMILES(False). Defaults to False.
        
    Returns:
        pd.Series: Tokenized dataset
    """
    output = np.array(
        [
            tokenize_with_space(tokenizer, sample, selfies)
            for sample in tqdm(dataset)
        ]
    )
    return output

def tokenize_sample_to_ids(tokenizer, sample):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(str(sample)).input_ids)
    return " ".join(tokens)

def tokenize_to_ids(tokenizer, samples):
    return np.array(
        [tokenize_sample_to_ids(tokenizer, sample) for sample in tqdm(samples)]
    )

if __name__ == "__main__":
    SMILES = pd.read_csv(
        PROCESSED_PATH / "isomers" / "full_deduplicated_isomers.csv", usecols=["SMILES"]
    ).values
    trained_SMILES_tokenizer = train_sentencepiece(
        SMILES, TOKENIZER_PATH / "smiles_trained_isomers", vocab_size=1000
    )
    atom_SMILES_tokenizer = train_atomwise_tokenizer(
        SMILES, TOKENIZER_PATH / "smiles_atom_isomers", vocab_size=1000
    )
    # with more rows, then RUST code of transformers panics
    SELFIES = pd.read_csv(
        PROCESSED_PATH / "isomers" / "full_deduplicated_isomers.csv",
        usecols=["SELFIES"],
        nrows=10000000,
    ).to_numpy(na_value="[None]")
    trained_SELFIES_tokenizer = train_sentencepiece(
        SELFIES, TOKENIZER_PATH / "selfies_trained_isomers", vocab_size=1000
    )
    atom_SELFIES_tokenizer = train_atomwise_tokenizer(
        SELFIES, TOKENIZER_PATH / "selfies_atom_isomers", vocab_size=1000
    )
    SMILES = pd.read_csv(
        PROCESSED_PATH / "standard" / "full_deduplicated_standard.csv",
        usecols=["SMILES"],
    ).values
    trained_SMILES_tokenizer = train_sentencepiece(
        SMILES, TOKENIZER_PATH / "smiles_trained_standard", vocab_size=1000
    )
    atom_SMILES_tokenizer = train_atomwise_tokenizer(
        SMILES, TOKENIZER_PATH / "smiles_atom_standard", vocab_size=1000
    )
    SELFIES = pd.read_csv(
        PROCESSED_PATH / "standard" / "full_deduplicated_standard.csv",
        usecols=["SELFIES"],
    ).values
    atom_SELFIES_tokenizer = train_atomwise_tokenizer(
        SELFIES, TOKENIZER_PATH / "selfies_atom_standard", vocab_size=1000
    )
    trained_SELFIES_tokenizer = train_sentencepiece(
        SELFIES, TOKENIZER_PATH / "selfies_trained_standard", vocab_size=1000
    )
