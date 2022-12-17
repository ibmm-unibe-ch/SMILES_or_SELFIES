""" BART Model creation
SMILES or SELFIES, 2022
"""
import logging

from transformers import BartConfig, BartModel


def get_BART_model(
    vocab_size: int = 50265,
    encoder_layers: int = 12,
    encoder_attention_heads: int = 16,
    decoder_layers: int = 12,
    decoder_attention_heads: int = 16,
    dropout: float = 0.1,
) -> BartModel:
    """Create empty BART model with given variables
    All default variables are the default from Transformers

    Args:
        vocab_size (int, optional): size of vocabulary. Defaults to 50265.
        encoder_layers (int, optional): Amount of encoder layers. Defaults to 12.
        encoder_attention_heads (int, optional): Amount of encoder attention heads. Defaults to 16.
        decoder_layers (int, optional): Amount of decoder layers. Defaults to 12.
        decoder_attention_heads (int, optional): Amount of decoder attention heads. Defaults to 16.
        dropout (float, optional): General dropout. Defaults to 0.1.

    Returns:
        BartModel: created BART model
    """
    config = BartConfig(
        vocab_size=vocab_size,
        encoder_layers=encoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        decoder_layers=decoder_layers,
        decoder_attention_heads=decoder_attention_heads,
        dropout=dropout,
    )
    logging.log("This is the BART config: {config}")
    return BartModel(config)
