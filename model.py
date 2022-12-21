""" BART Model creation
SMILES or SELFIES, 2022
"""
import logging

from transformers import BartConfig, BartModel


def get_BART_model(
    vocab_size: int = 50265,
    encoder_layers: int = 6,
    encoder_attention_heads: int = 12,
    decoder_layers: int = 6,
    decoder_attention_heads: int = 12,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    classif_dropout: float = 0.1,
    classifier_dropout: float = 0.0,
    d_model: int = 768,
    decoder_ffn_dim: int = 3072,
    early_stopping: bool = True,
    encoder_ffn_dim: int = 3072,
    encoder_layerdrop: float = 0.0,
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
        attention_dropout=attention_dropout,
        classif_dropout=classif_dropout,
        classifier_dropout=classifier_dropout,
        d_model=d_model,
        decoder_ffn_dim=decoder_ffn_dim,
        early_stopping=early_stopping,
        encoder_ffn_dim=encoder_ffn_dim,
        encoder_layerdrop=encoder_layerdrop,
    )
    logging.info("This is the BART config: {config}")
    return BartModel(config)
