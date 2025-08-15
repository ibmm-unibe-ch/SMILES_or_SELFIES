""" Tokenisation
SMILES or SELFIES, 2022
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from rdkit import RDLogger, Chem
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

from constants import PROCESSED_PATH, TOKENIZER_PATH
from preprocessing import canonize_smile, translate_selfie

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.warning")

# Special tokens for all tokenizers
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
# Default vocabulary size
DEFAULT_VOCAB_SIZE = 1000

def train_sentencepiece(
    training_data: pd.Series,
    save_path: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE
) -> BartTokenizerFast:
    """Train a SentencePiece tokenizer on molecular representations.

    Args:
        training_data: Series containing SMILES or SELFIES strings for training.
        save_path: Directory to save the trained tokenizer.
        vocab_size: Size of the vocabulary to generate. Defaults to 1000.

    Returns:
        A BartTokenizerFast instance wrapping the trained tokenizer.

    Example:
        >>> tokenizer = train_sentencepiece(smiles_series, Path("tokenizers/smiles"))
    """
    tk_tokenizer = SentencePieceUnigramTokenizer()
    tk_tokenizer.train_from_iterator(
        training_data,
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
    )
    
    tokenizer = BartTokenizerFast(tokenizer_object=tk_tokenizer)
    logging.info(f"Saving SentencePiece tokenizer to {save_path}")
    tokenizer.save_pretrained(save_path)
    return tokenizer


def train_atomwise_tokenizer(
    training_data: pd.Series,
    save_path: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE
) -> BartTokenizerFast:
    """Train an atom-wise tokenizer for molecular representations.

    Uses regex-based splitting to tokenize at the atom/bond level.

    Args:
        training_data: Series containing SMILES or SELFIES strings for training.
        save_path: Directory to save the trained tokenizer.
        vocab_size: Size of the vocabulary to generate. Defaults to 1000.

    Returns:
        A BartTokenizerFast instance wrapping the trained tokenizer.

    Example:
        >>> tokenizer = train_atomwise_tokenizer(smiles_series, Path("tokenizers/atomwise"))
    """
    tk_tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    # copied from https://colab.research.google.com/drive/1tsiTpC4i26QNdRzBHFfXIOFVToE54-9b?usp=sharing#scrollTo=UHzrWuFpCtzs
    # same in DeepChem
    # Regex pattern for atom-wise tokenization
    splitting_regex = Regex(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    )
    tk_tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=splitting_regex, behavior="isolated"
    )
    
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS
    )
    
    tk_tokenizer.train_from_iterator(training_data, trainer=trainer)
    tokenizer = BartTokenizerFast(tokenizer_object=tk_tokenizer)
    logging.info(f"Saving atom-wise tokenizer to {save_path}")
    tokenizer.save_pretrained(save_path)
    return tokenizer


def get_tokenizer(tokenizer_path: Path) -> BartTokenizerFast:
    """Load a pretrained tokenizer from disk.

    Args:
        tokenizer_path: Directory containing the saved tokenizer.

    Returns:
        Loaded BartTokenizerFast instance.

    Raises:
        OSError: If the tokenizer cannot be loaded from the specified path.
    """
    try:
        return BartTokenizerFast.from_pretrained(tokenizer_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer from {tokenizer_path}: {str(e)}")
        raise


def tokenize_with_space(
    tokenizer: BartTokenizerFast,
    molecule: str,
    selfies: bool = False,
    big_c: bool = False
) -> Optional[str]:
    """Tokenize a single molecule representation.

    Args:
        tokenizer: Tokenizer to use for tokenization.
        molecule: SMILES or SELFIES string to tokenize.
        selfies: Whether to convert to SELFIES before tokenization.
        big_c: Whether to kekulize aromatic bonds (for SMILES only).

    Returns:
        Space-separated tokens as a string, or None if molecule is invalid.

    Example:
        >>> tokens = tokenize_single_molecule(tokenizer, "CCO", use_selfies=False)
    """
    # Check if molecule can be translated to SELFIES if requested
    if selfies and translate_selfie(str(molecule))[0] is None:
        return None

    # Canonicalize SMILES
    canon_smiles = canonize_smile(molecule)
    if not canon_smiles:
        return None

    # Kekulize if requested
    if big_c:
        mol = Chem.MolFromSmiles(canon_smiles)
        if mol:
            canon_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

    # Convert to SELFIES if requested
    if selfies:
        canon_smiles = translate_selfie(str(canon_smiles))[0]

    # Tokenize and return space-separated tokens
    token_ids = tokenizer(str(canon_smiles)).input_ids
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return " ".join(tokens)


def tokenize_dataset(
    tokenizer: BartTokenizerFast,
    dataset: pd.Series,
    use_selfies: bool = False,
    big_c: bool = False
) -> np.ndarray:
    """Tokenize a dataset of molecular representations.

    Args:
        tokenizer: Tokenizer to use for tokenization.
        dataset: Series containing molecules to tokenize.
        use_selfies: Whether to convert to SELFIES before tokenization.
        big_c: Whether to kekulize aromatic bonds (for SMILES only).

    Returns:
        Array of tokenized molecules (space-separated strings).

    Example:
        >>> tokenized = tokenize_dataset(tokenizer, smiles_series, use_selfies=False)
    """
    return np.array([
        tokenize_with_space(tokenizer, sample, use_selfies, big_c)
        for sample in tqdm(dataset, desc="Tokenizing dataset")
    ])


def tokenize_to_ids(
    tokenizer: BartTokenizerFast,
    samples: Union[pd.Series, list]
) -> np.ndarray:
    """Tokenize samples to space-separated token strings.

    Args:
        tokenizer: Tokenizer to use for tokenization.
        samples: Molecular representations to tokenize.

    Returns:
        Array of space-separated token strings.
    """
    return np.array([
        " ".join(tokenizer.convert_ids_to_tokens(tokenizer(str(sample)).input_ids))
        for sample in tqdm(samples, desc="Tokenizing to IDs")
    ])


def main() -> None:
    """Main function to train and save tokenizers for different configurations."""
    # SMILES tokenizers for isomers dataset
    smiles_isomers = pd.read_csv(
        PROCESSED_PATH / "isomers" / "full_deduplicated_isomers.csv",
        usecols=["SMILES"]
    ).values
    
    train_sentencepiece(
        smiles_isomers,
        TOKENIZER_PATH / "smiles_trained_isomers"
    )
    train_atomwise_tokenizer(
        smiles_isomers,
        TOKENIZER_PATH / "smiles_atom_isomers"
    )

    # SELFIES tokenizers for isomers dataset (limited to 10M rows)
    selfies_isomers = pd.read_csv(
        PROCESSED_PATH / "isomers" / "full_deduplicated_isomers.csv",
        usecols=["SELFIES"],
        nrows=10_000_000
    ).to_numpy(na_value="[None]")
    
    train_sentencepiece(
        selfies_isomers,
        TOKENIZER_PATH / "selfies_trained_isomers"
    )
    train_atomwise_tokenizer(
        selfies_isomers,
        TOKENIZER_PATH / "selfies_atom_isomers"
    )

    # SMILES tokenizers for standard dataset
    smiles_standard = pd.read_csv(
        PROCESSED_PATH / "standard" / "full_deduplicated_standard.csv",
        usecols=["SMILES"]
    ).values
    
    train_sentencepiece(
        smiles_standard,
        TOKENIZER_PATH / "smiles_trained_standard"
    )
    train_atomwise_tokenizer(
        smiles_standard,
        TOKENIZER_PATH / "smiles_atom_standard"
    )

    # SELFIES tokenizers for standard dataset
    selfies_standard = pd.read_csv(
        PROCESSED_PATH / "standard" / "full_deduplicated_standard.csv",
        usecols=["SELFIES"]
    ).values
    
    train_sentencepiece(
        selfies_standard,
        TOKENIZER_PATH / "selfies_trained_standard"
    )
    train_atomwise_tokenizer(
        selfies_standard,
        TOKENIZER_PATH / "selfies_atom_standard"
    )


if __name__ == "__main__":
    main()