"""
Utility functions for working with Fairseq models and SMILES/SELFIES molecular data.

Includes:
    - Model loading and manipulation
    - Dataset handling
    - Prediction and evaluation helpers
    - Token dictionary utilities
    - Embedding extraction and attention weight analysis
    - Preprocessing helpers for SMILES/SELFIES data
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.roberta import RobertaModel
from fairseq.models.bart import BARTModel, BARTHubInterface

from constants import (
    PARSING_REGEX,
    TASK_PATH,
    MODEL_PATH,
    PREDICTION_MODEL_PATH,
    FAIRSEQ_PREPROCESS_PATH,
    TOKENIZER_SUFFIXES,
    TOKENIZER_PATH,
)
from tokenisation import get_tokenizer, tokenize_dataset

os.environ["MKL_THREADING_LAYER"] = "GNU"


# ---------------------------------------------------------------------------
# Model Loading and Dataset Handling
# ---------------------------------------------------------------------------

def load_model(
    model_path: Path,
    data_path: Path,
    cuda: Optional[int] = None

) -> Union[RobertaModel, BARTHubInterface]:
    """
    Load a Fairseq BART or RoBERTa model.

    Args:
        model_path: Path to the .pt model checkpoint.
        data_path: Path to the Fairseq-preprocessed data directory.
        cuda: CUDA device index. If None, stays on CPU.

    Returns:
        Fairseq model in evaluation mode.
    """
    if "roberta" in str(model_path).lower():
        model = RobertaModel.from_pretrained(
            str(model_path.parent),
            data_name_or_path=str(data_path),
            checkpoint_file=model_path.name,
        )
    else:
        model = BARTModel.from_pretrained(
            str(model_path.parent),
            data_name_or_path=str(data_path),
            checkpoint_file=model_path.name,
        )

    model.eval()
    if cuda is not None:
        model.cuda(device=f"cuda:{cuda}")
    return model


def load_dataset(
    data_path: Path,
    classification: bool = True
) -> List[Union[str, float]]:
    """
    Load a Fairseq dataset.

    Args:
        data_path (Path): folder path of data (e.g. /input0/test)
            for classification: TASK_PATH / task / tokenizer / "label" / "test" , 
            for regression: TASK_PATH / task / tokenizer / "label" / "test.label"
        classification (bool): if classification(True) or regression(False) loading should be used. Defaults to classification.


    Returns:
        List of dataset items (tokens or float labels).
    """
    if classification:
        dictionary = Dictionary.load(str(data_path.parent / "dict.txt"))
        return list(load_indexed_dataset(str(data_path), dictionary))

    with open(data_path, "r") as f:
        return [float(line.strip()) for line in f]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def get_predictions(
    model: Union[RobertaModel, BARTHubInterface],
    mols: np.ndarray,
    targets: np.ndarray,
    target_dict_path: Optional[Path],
    classification: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Get predictions for molecules using a Fairseq model.

    Args:
        model: Fairseq model.
        mols: Tokenized molecule dataset.
        targets: Target labels or regression values.
        target_dict_path: Path to Fairseq target dictionary.
        classification: If True, classification mode; else regression.

    Returns:
        Tuple of (predictions, true_targets).
    """
    preds, seen_targets = [], []
    target_dict = Dictionary.load(str(target_dict_path)) if classification else None

    for smile, target in tqdm(zip(mols, targets), total=len(mols)):
        smile_tensor = torch.cat((torch.tensor([0]), smile[:126], torch.tensor([2])))
        output = model.predict(
            "sentence_classification_head",
            smile_tensor,
            return_logits=not classification,
        )

        if classification:
            target_val = target[0].item()
            if target_dict[4] == "1":
                preds.append(output[0][0].exp().item())
                seen_targets.append(-1 * target_val + 5)
            else:
                preds.append(output[0][1].exp().item())
                seen_targets.append(target_val - 4)
        else:
            preds.append(output[0][0].item())
            seen_targets.append(target)

    return preds, seen_targets


# ---------------------------------------------------------------------------
# Dictionary Utilities
# ---------------------------------------------------------------------------

def create_dict_from_fairseq(
    fairseq_dict_file: Path,
    output_path: Path
) -> Dict[str, int]:
    """
    Convert a Fairseq dictionary file into JSON format.

    Args:
        fairseq_dict_file: Path to Fairseq dict.txt.
        output_path: Path to save the JSON dictionary.

    Returns:
        Dictionary mapping tokens to IDs.
    """
    output = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    with open(fairseq_dict_file, "r") as f:
        for idx, line in enumerate(f):
            token = line.split()[0].strip()
            output[token] = len(output) + idx

    with open(output_path, "w") as f:
        json.dump(output, f)
    return output


def get_dictionary(target_dict_path: Path) -> Dictionary:
    """
    Load a Fairseq dictionary.

    Args:
        target_dict_path: Path to dict.txt.

    Returns:
        Fairseq Dictionary object.
    """
    return Dictionary.load(str(target_dict_path))


# ---------------------------------------------------------------------------
# Model Manipulation
# ---------------------------------------------------------------------------

def transplant_model(
    taker_model_path: Path,
    giver_model_path: Path,
    output_path: Optional[Path] = None
) -> None:
    """
    Replace parameters in taker model with those from giver model.

    Args:
        taker_model_path: Path to model receiving parameters.
        giver_model_path: Path to model donating parameters.
        output_path: Path to save new model. Defaults to taker path.
    """
    taker_model = torch.load(taker_model_path)
    giver_model = torch.load(giver_model_path)
    output_path = output_path or taker_model_path

    for key, value in giver_model["model"].items():
        taker_model["model"][key] = value

    torch.save(taker_model, output_path)
    logging.info(f"Transplanted {giver_model_path} into {output_path}")


def transform_to_prediction_model(suffix: str) -> None:
    """
    Convert a diffusion model into a Fairseq prediction model.

    Args:
        suffix: Model suffix identifying tokenizer and architecture.
    """
    tokenizer_suffix = "_".join(suffix.split("_")[:-1])
    architecture = "bart_base" if "bart" in suffix else "roberta_base"

    cmd = (
        f'CUDA_VISIBLE_DEVICES=0 fairseq-train {TASK_PATH/"lipo"/tokenizer_suffix} '
        f'--update-freq 1 '
        f'--restore-file {MODEL_PATH/suffix/"checkpoint_last.pt"} '
        f'--batch-size 1 '
        f'--task sentence_prediction --num-workers 1 --add-prev-output-tokens '
        f'--layernorm-embedding --reset-optimizer --reset-dataloader --reset-meters '
        f'--required-batch-size-multiple 1 --arch {architecture} '
        f'--skip-invalid-size-inputs-valid-test --criterion sentence_prediction '
        f'--optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 0 --weight-decay 0.01 '
        f'--clip-norm 0.1 --lr 0.0 --max-update 1 --warmup-updates 1 '
        f'--keep-best-checkpoints 1 --num-classes 1 --save-dir {PREDICTION_MODEL_PATH/suffix} '
        f'--best-checkpoint-metric loss --regression-target --init-token 0'
    )
    if architecture == "bart_base":
        cmd += " --share-all-embeddings --share-decoder-input-output-embed --max-target-positions 1024"

    os.system(cmd)
    transplant_model(
        PREDICTION_MODEL_PATH / suffix / "checkpoint_last.pt",
        MODEL_PATH / suffix / "checkpoint_last.pt",
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_series_old(series: np.ndarray, output_path: Path, dictionary_path: Path):
    file_path = str(output_path) + ".csv"
    os.makedirs(output_path.parent, exist_ok=True)
    series.tofile(file_path, sep="\n", format="%s")
    os.system(
        (
            f"fairseq-preprocess --only-source --trainpref {file_path}  --destdir {output_path} --srcdict {dictionary_path} --workers 60"
        )
    )


def preprocess_series(
    series: pd.Series,
    output_path: Path,
    tokenizer_suffix: Optional[str] = None
) -> None:
    """
    Tokenize and preprocess a series of molecules for Fairseq.

    Args:
        series: Series of SMILES or SELFIES strings.
        output_path: Path to save preprocessed data.
        tokenizer_suffix: Specific tokenizer suffix to use. If None, process for all TOKENIZER_SUFFIXES.
    """
    suffixes = TOKENIZER_SUFFIXES if tokenizer_suffix is None else [tokenizer_suffix]

    for suffix in suffixes:
        output_dir = output_path / suffix
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = get_tokenizer(TOKENIZER_PATH / suffix)
        selfies = suffix.startswith("selfies")
        tokenized = tokenize_dataset(tokenizer, series, selfies)

        train_input = output_dir / "train.input"
        tokenized.tofile(train_input, sep="\n", format="%s")

        model_dict = FAIRSEQ_PREPROCESS_PATH / suffix / "dict.txt"
        dest_dir = output_dir / "input0"
        os.system(
            f'fairseq-preprocess --only-source --trainpref {train_input} --testpref {output_dir/"test.input "}'
            f'--destdir {dest_dir} --srcdict {model_dict} --workers 60'
        )


def create_random_prediction_model(output_path: Path) -> None:
    """
    Create a randomly initialized BART prediction model.

    Args:
        output_path: Path to save the generated model.
    """
    if output_path.exists():
        return

    # Step 1: Pre-train a random BART model
    os.system(
        f'CUDA_VISIBLE_DEVICES=0 micromamba run -n fairseq_git fairseq-train '
        f'{FAIRSEQ_PREPROCESS_PATH}/smiles_atom_isomers --save-dir {MODEL_PATH/"random_bart"} '
        f'--max-source-positions 1024 --batch-size 32 --mask 0.0 --tokens-per-sample 512 '
        f'--max-update 1 --warmup-updates 1 --task denoising --save-interval 1 '
        f'--arch bart_base --optimizer adam --lr 0.0 --dropout 0.0 --criterion cross_entropy '
        f'--max-tokens 3200 --weight-decay 0.0 --attention-dropout 0.0 --relu-dropout 0.0 '
        f'--share-decoder-input-output-embed --share-all-embeddings --clip-norm 1.0 '
        f'--skip-invalid-size-inputs-valid-test --log-format json --seed 4 '
        f'--distributed-world-size 1 --no-epoch-checkpoints --mask-length span-poisson '
        f'--encoder-learned-pos --decoder-learned-pos --rotate 0.0 --mask-random 0.0 '
        f'--insert 0.0 --poisson-lambda 3.5 --dataset-impl mmap --num-workers 4'
    )

    # Step 2: Fine-tune as a prediction model
    os.system(
        f'CUDA_VISIBLE_DEVICES=0 fairseq-train {TASK_PATH/"lipo"/"smiles_atom_isomers"} '
        f'--restore-file {MODEL_PATH/"random_bart"/"checkpoint_last.pt"} --save-dir {output_path} '
        f'--max-source-positions 1024 --update-freq 1 --batch-size 1 --task sentence_prediction '
        f'--num-workers 1 --layernorm-embedding --share-all-embeddings '
        f'--share-decoder-input-output-embed --required-batch-size-multiple 1 '
        f'--add-prev-output-tokens --reset-optimizer --reset-dataloader --reset-meters '
        f'--reset-lr-scheduler --arch bart_base --skip-invalid-size-inputs-valid-test '
        f'--criterion sentence_prediction --max-target-positions 1024 --optimizer adam '
        f'--adam-betas "(0.9, 0.999)" --adam-eps 0.1 --clip-norm 0.1 --lr 0.0 '
        f'--max-update 2 --warmup-updates 1 --keep-best-checkpoints 1 --num-classes 1 '
        f'--best-checkpoint-metric loss --regression-target --init-token 0'
    )


# ---------------------------------------------------------------------------
# Embedding & Attention Functions
# ---------------------------------------------------------------------------

def generate_prev_output_tokens(
    sample: torch.Tensor,
    source_dictionary: Dictionary
) -> torch.Tensor:
    """
    Generate previous output tokens for Fairseq models.

    Args:
        sample: Tokenized sample as a tensor (1D).
        source_dictionary: Fairseq dictionary for padding index lookup.

    Returns:
        Tensor of previous output tokens.
    """
    tokens = sample.unsqueeze(0)
    prev_output_tokens = tokens.clone()
    prev_output_tokens[:, 0] = tokens.gather(
        1, (tokens.ne(source_dictionary.pad()).sum(1) - 1).unsqueeze(-1)
    ).squeeze()
    prev_output_tokens[:, 1:] = tokens[:, :-1]
    return prev_output_tokens


def get_embeddings(
    model: Union[RobertaModel, BARTHubInterface],
    dataset: Union[Path, str, List[torch.Tensor]],
    source_dictionary: Dictionary,
    whole_mol: bool = True,
    cuda: int = 0
) -> List[np.ndarray]:
    """
    Extract embeddings for a dataset from a Fairseq model.

    Args:
        model: Fairseq model.
        dataset: Path to dataset, string path, or list of tokenized samples.
        source_dictionary: Fairseq dictionary for decoding.
        whole_mol: If True, return only last-token (molecule-level) embedding.
        cuda: CUDA device index.

    Returns:
        List of embeddings (numpy arrays).
    """
    if isinstance(dataset, (Path, str)):
        dataset = list(load_indexed_dataset(str(dataset), source_dictionary))

    embeddings = []
    for sample in tqdm(dataset):
        sample = sample[:1020]
        if isinstance(model, BARTHubInterface):
            prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(f"cuda:{cuda}")
            features = model.model(
                sample.unsqueeze(0).to(f"cuda:{cuda}"),
                None,
                prev_output_tokens,
                features_only=True,
            )[0][0]
        else:
            features = model.model(
                sample.unsqueeze(0).to(f"cuda:{cuda}"),
                classification_head_name=None, features_only=True,
            )[0][0]

        features = features.detach().cpu().numpy()
        embeddings.append(features[-1, :] if whole_mol else features[:-1, :])
    return embeddings


def compute_attention_output(
    model: Union[RobertaModel, BARTHubInterface],
    texts: List[str],
    source_dictionary: Dictionary,
    tokenizer: Optional[Any] = None
) -> List[List[Tuple[float, str]]]:
    """
    Compute attention weights for a list of input texts.
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py

    Args:
        model: Fairseq model.
        texts: List of SMILES/SELFIES strings.
        source_dictionary: Fairseq dictionary.
        tokenizer: Optional HuggingFace tokenizer.

    Returns:
        List of attention-weight lists, each containing (weight, token) tuples.
    """
    device = next(model.parameters()).device
    dataset_attentions = []

    for text in texts:
        if tokenizer is None:
            parsed_tokens = [tok for tok in re.split(PARSING_REGEX, text) if tok]
        else:
            parsed_tokens = tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids)
        sample = torch.tensor(tokenizer(text).input_ids)

        prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(device)
        _, extra = model.model(
            sample.unsqueeze(0).to(device),
            None,
            prev_output_tokens,
            features_only=False,
            return_all_hiddens=False
        )

        token_attention = extra["attn"][0][0][-1].cpu().detach().tolist()
        dataset_attentions.append(list(zip(token_attention, parsed_tokens)))
        # Attention does not add up to 1, because [CLS] token at the end takes a lot of attention
    return dataset_attentions


def compute_embedding_output(
    model: Union[RobertaModel, BARTHubInterface],
    texts: List[str],
    source_dictionary: Dictionary,
    tokenizer: Optional[Any] = None,
    cuda: int = 0
) -> List[List[Tuple[List[float], str]]]:
    """
    Compute token-level embeddings for a list of input texts.
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py

    Args:
        model: Fairseq model.
        texts: List of SMILES/SELFIES strings.
        source_dictionary: Fairseq dictionary.
        tokenizer: Optional HuggingFace tokenizer.
        cuda: CUDA device index.

    Returns:
        List of token-embedding lists, each containing (embedding_vector, token) tuples.
    """
    device = next(model.parameters()).device
    dataset_embeddings = []

    for text in texts:
        if tokenizer is None:
            parsed_tokens = [tok for tok in re.split(PARSING_REGEX, text) if tok]
        else:
            parsed_tokens = tokenizer.convert_ids_to_tokens(tokenizer(str(text)).input_ids)
        sample = torch.tensor(tokenizer(text).input_ids)

        token_embeddings, _ = model.model(sample.unsqueeze(0).to(device), None, classification_head_name=None, features_only=True)
        token_embeddings_list = token_embeddings[0].cpu().detach().tolist()
        dataset_embeddings.append(list(zip(token_embeddings_list, parsed_tokens)))
    return dataset_embeddings


def compute_model_output(
    dataset: List[torch.Tensor],
    model: Union[RobertaModel, BARTHubInterface],
    texts: List[str],
    source_dictionary: Dictionary,
    attentions: bool = True,
    eos_attentions: bool = True,
    embeddings: bool = False,
    eos_embedding: bool = False,
    tokenizer: Optional[Any] = None
) -> Tuple[
    Optional[List[List[Tuple[float, str]]]],
    Optional[List[List[Tuple[float, str]]]],
    Optional[List[List[Tuple[List[float], str]]]],
    Optional[List[List[float]]]
]:
    """
    Compute attentions and/or embeddings for a dataset.

    Args:
        dataset: Pre-tokenized dataset (list of tensors).
        model: Fairseq model.
        texts: Human-readable SMILES/SELFIES strings.
        source_dictionary: Fairseq dictionary.
        attentions: If True, return per-token attention weights.
        eos_attentions: If True, return attention weights for EOS token.
        embeddings: If True, return per-token embeddings.
        eos_embedding: If True, return EOS token embedding.
        tokenizer: Optional HuggingFace tokenizer.

    Returns:
        Tuple of (attentions, eos_attentions, embeddings, eos_embeddings).
    """
    device = next(model.parameters()).device
    dataset_attentions = [] if attentions else None
    dataset_eos_attentions = [] if eos_attentions else None
    dataset_embeddings = [] if embeddings else None
    dataset_eos_embeddings = [] if eos_embedding else None

    for sample, text in zip(dataset, texts):
        if tokenizer is None:
            parsed_tokens = [tok for tok in re.split(PARSING_REGEX, text) if tok]
        else:
            parsed_tokens = tokenizer.convert_ids_to_tokens(tokenizer(str(text)).input_ids)

        prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(device)
        token_embeddings_tensor, extra = model.model(
            sample.unsqueeze(0).to(device),
            None,
            prev_output_tokens,
            features_only=False
        )

        if attentions:
            attention_matrix = extra["attn"][0][0].cpu().detach().tolist()
            dataset_attentions.append(list(zip(attention_matrix, parsed_tokens)))

        if eos_attentions:
            eos_attention_vector = extra["attn"][0][0][-1].cpu().detach().tolist()
            dataset_eos_attentions.append(list(zip(eos_attention_vector, parsed_tokens)))

        if embeddings:
            token_emb_list = token_embeddings_tensor[0].cpu().detach().tolist()
            dataset_embeddings.append(list(zip(token_emb_list, parsed_tokens)))

        if eos_embedding:
            dataset_eos_embeddings.append(token_embeddings_tensor[0][-1].cpu().detach().tolist())

    return dataset_attentions, dataset_eos_attentions, dataset_embeddings, dataset_eos_embeddings