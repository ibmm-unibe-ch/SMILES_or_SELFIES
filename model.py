""" BART Model creation
SMILES or SELFIES, 2022
"""
from transformers import BartModel, BartConfig
import logging


def get_BART_model(
    vocab_size: int = 50265,
    encoder_layers: int = 12,
    encoder_attention_heads: int = 16,
    decoder_layers: int = 12,
    decoder_attention_heads: int = 16,
    dropout: float = 0.1,
) -> BartModel:
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
