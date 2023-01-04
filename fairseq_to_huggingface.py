""" Translate fairseq model to huggingface
steal config.json from bart-base and put it in bart_base, adjust vocab size
adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py
seen in https://discuss.huggingface.co/t/how-can-i-convert-a-model-created-with-fairseq/564
SMILES or SELFIES, 2022
"""

VOCAB_SIZE = 51201
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from transformers.utils import logging

import fairseq

FAIRSEQ_MODELS = [
    "bart.large",
    "bart.large.mnli",
    "bart.large.cnn",
    "bart_xsum/model.pt",
    "bart.base",
]
extra_arch = {
    "bart.large": BartModel,
    "bart.base": BartModel,
    "bart.large.mnli": BartForSequenceClassification,
}


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "c c c c c"

mnli_rename_keys = [
    (
        "model.classification_heads.mnli.dense.weight",
        "classification_head.dense.weight",
    ),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    (
        "model.classification_heads.mnli.out_proj.weight",
        "classification_head.out_proj.weight",
    ),
    (
        "model.classification_heads.mnli.out_proj.bias",
        "classification_head.out_proj.bias",
    ),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "encoder.layers.0.in_proj_weight",
        "encoder.layers.0.in_proj_bias",
        "encoder.layers.0.out_proj_weight",
        "encoder.layers.0.out_proj_bias",
        "encoder.layers.0.fc1_weight",
        "encoder.layers.0.fc1_bias",
        "encoder.layers.0.fc2_weight",
        "encoder.layers.0.fc2_bias",
        "encoder.layers.1.in_proj_weight",
        "encoder.layers.1.in_proj_bias",
        "encoder.layers.1.out_proj_weight",
        "encoder.layers.1.out_proj_bias",
        "encoder.layers.1.fc1_weight",
        "encoder.layers.1.fc1_bias",
        "encoder.layers.1.fc2_weight",
        "encoder.layers.1.fc2_bias",
        "encoder.layers.2.in_proj_weight",
        "encoder.layers.2.in_proj_bias",
        "encoder.layers.2.out_proj_weight",
        "encoder.layers.2.out_proj_bias",
        "encoder.layers.2.fc1_weight",
        "encoder.layers.2.fc1_bias",
        "encoder.layers.2.fc2_weight",
        "encoder.layers.2.fc2_bias",
        "encoder.layers.3.in_proj_weight",
        "encoder.layers.3.in_proj_bias",
        "encoder.layers.3.out_proj_weight",
        "encoder.layers.3.out_proj_bias",
        "encoder.layers.3.fc1_weight",
        "encoder.layers.3.fc1_bias",
        "encoder.layers.3.fc2_weight",
        "encoder.layers.3.fc2_bias",
        "encoder.layers.4.in_proj_weight",
        "encoder.layers.4.in_proj_bias",
        "encoder.layers.4.out_proj_weight",
        "encoder.layers.4.out_proj_bias",
        "encoder.layers.4.fc1_weight",
        "encoder.layers.4.fc1_bias",
        "encoder.layers.4.fc2_weight",
        "encoder.layers.4.fc2_bias",
        "encoder.layers.5.in_proj_weight",
        "encoder.layers.5.in_proj_bias",
        "encoder.layers.5.out_proj_weight",
        "encoder.layers.5.out_proj_bias",
        "encoder.layers.5.fc1_weight",
        "encoder.layers.5.fc1_bias",
        "encoder.layers.5.fc2_weight",
        "encoder.layers.5.fc2_bias",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    sd["model"]["encoder.embed_tokens.weight"] = F.pad(
        input=sd["model"]["encoder.embed_tokens.weight"],
        pad=(
            0,
            0,
            0,
            VOCAB_SIZE - sd["model"]["encoder.embed_tokens.weight"].size()[0],
        ),
    )
    sd["model"]["decoder.embed_tokens.weight"] = F.pad(
        input=sd["model"]["decoder.embed_tokens.weight"],
        pad=(
            0,
            0,
            0,
            VOCAB_SIZE - sd["model"]["decoder.embed_tokens.weight"].size()[0],
        ),
    )
    sd["model"]["decoder.output_projection.weight"] = F.pad(
        input=sd["model"]["decoder.output_projection.weight"],
        pad=(
            0,
            0,
            0,
            51201 - sd["model"]["decoder.output_projection.weight"].size()[0],
        ),
    )
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.base").eval()
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_bart_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None
):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        bart = load_xsum_checkpoint(checkpoint_path)

    bart.model.upgrade_state_dict(bart.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    config = BartConfig.from_pretrained(hf_checkpoint_name)
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)

    if checkpoint_path == "bart.large.mnli":
        state_dict = bart.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["model.shared.weight"] = state_dict[
            "model.decoder.embed_tokens.weight"
        ]
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        model = BartForSequenceClassification(config).eval()
        model.load_state_dict(state_dict)
        fairseq_output = bart.predict("mnli", tokens, return_logits=True)
        new_model_outputs = model(tokens)[0]  # logits
    else:  # no classification heads to worry about
        state_dict = bart.model.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        fairseq_output = bart.extract_features(tokens)
        model = BartModel(config).eval()
        model.load_state_dict(state_dict)
        new_model_outputs = model(tokens)

    # Check results
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path",
        type=str,
        help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem.",
    )
    parser.add_argument(
        "pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--hf_config",
        default=None,
        type=str,
        help="Which huggingface architecture to use: bart-large-xsum",
    )
    args = parser.parse_args()
    convert_bart_checkpoint(
        args.fairseq_path,
        args.pytorch_dump_folder_path,
        hf_checkpoint_name=args.hf_config,
    )
