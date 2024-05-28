"""Util functions to deal with fairseq
SMILES or SELFIES, 2023
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from constants import PARSING_REGEX, PROJECT_PATH, TASK_PATH, MODEL_PATH, PREDICTION_MODEL_PATH, FAIRSEQ_PREPROCESS_PATH
from tqdm import tqdm

from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.roberta import RobertaModel
from fairseq.models.bart import BARTModel, BARTHubInterface


os.environ["MKL_THREADING_LAYER"] = "GNU"


def load_model(model_path: Path, data_path: Path, cuda_device: str = None):
    """Load fairseq BART model

    Args:
        model_path (Path): path to .pt file
        cuda_device (str, optional): if model should be converted to a device. Defaults to None.

    Returns:
        fairseq_model: load BART model
    """
    if "roberta" in str(model_path):
        model = RobertaModel.from_pretrained(
            str(model_path.parent),
            data_name_or_path=str(data_path),
            checkpoint_file=str(model_path.name),
        )
    else:
        model = BARTModel.from_pretrained(
            str(model_path.parent),
            data_name_or_path=str(data_path),
            checkpoint_file=str(model_path.name),
        )
    model.eval()
    if cuda_device:
        model.cuda(device=str(f"cuda:{str(cuda_device)}"))
    return model


def load_dataset(data_path: Path, classification: bool = True) -> List[str]:
    """Load dataset with fairseq

    Args:
        data_path (Path): folder path of data (e.g. /input0/test)
        classification (bool): if classification(True) or regression(False) loading should be used. Defaults to classification.


    Returns:
        List[str]: loaded fairseq dataset
    """
    if classification:
        dikt = Dictionary.load(str(data_path.parent / "dict.txt"))
        data = list(load_indexed_dataset(str(data_path), dikt))
        return data
    with open(data_path, "r") as label_file:
        label_lines = label_file.readlines()
    return [float(line.strip()) for line in label_lines]


def get_predictions(
    model,
    mols: np.ndarray,
    targets: np.ndarray,
    target_dict_path: Path,  # maybe None for classifications?
    classification: bool = True,
) -> Tuple[List[float], List[float]]:
    """Get predictions of model on mols

    Args:
        model (fairseq_model): fairseq model to make predictions with
        mols (np.ndarray): dataset to make predictions
        targets (np.ndarray): targets to predict against
        target_dict_path (Path): path to target_dict to translate model output to class
        classification (bool): if classification(True) or regression(False). Defaults to classification.

    Returns:
        Tuple[List[float], List[float]]: predictions, targets
    """
    # from https://github.com/YerevaNN/BARTSmiles/blob/main/evaluation/compute_score.py
    preds = []
    seen_targets = []
    if classification:
        target_dict = Dictionary.load(str(target_dict_path))
    for (smile, target) in tqdm(list(zip(mols, targets))):
        smile = torch.cat(
            (torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2]))
        )
        output = model.predict(
            "sentence_classification_head", smile, return_logits=not classification
        )
        if classification:
            target = target[0].item()
            if target_dict[4] == "1":
                preds.append(output[0][0].exp().item())
                seen_targets.append(-1 * target + 5)
            else:
                preds.append(output[0][1].exp().item())
                seen_targets.append(target - 4)
        else:
            preds.append(output[0][0].item())
            seen_targets.append(target)
    return preds, seen_targets


def create_dict_from_fairseq(fairseq_dict_dir: Path, output_path: Path):
    # hopefully never needed
    dict_stub = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    output = dict_stub
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


def transplant_model(
    taker_model_path: Path, giver_model_path: Path, output_path: Optional[Path] = None
):
    """Transplanting a diffusion (giver) model to a different (taker) model and save it to output_path

    Args:
        taker_model_path (Path): path to taker model
        giver_model_path (Path): path to giver model
        output_path (Optional[Path], optional): path where to save transplanted model to. Defaults to taker model.
    """
    taker_model = torch.load(taker_model_path)
    giver_model = torch.load(giver_model_path)
    if output_path is None:
        output_path = taker_model_path
    for key, value in giver_model["model"].items():
        taker_model["model"][key] = value
    torch.save(taker_model, output_path)
    logging.info(
        f"Transplanted {giver_model_path} into {output_path} using {taker_model_path}"
    )


def transform_to_prediction_model(suffix):
    """Transform a diffusion model to a prediction model, which then can be used by fairseq."""
    tokenizer_suffix = "_".join(re.split("_", suffix)[:-1])
    architecture = "bart_base" if "bart" in suffix else "roberta_base"
    if architecture == "roberta_base":
        os.system(
            f'CUDA_VISIBLE_DEVICES=0 fairseq-train {TASK_PATH/"lipo"/tokenizer_suffix} --update-freq 1 --restore-file {MODEL_PATH/suffix/"checkpoint_last.pt"} --batch-size 1 --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch {architecture} --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 0 --weight-decay 0.01 --clip-norm 0.1 --lr 0.0 --max-update 1 --warmup-updates 1 --keep-best-checkpoints 1 --num-classes 1 --save-dir {PREDICTION_MODEL_PATH/suffix} --best-checkpoint-metric loss --regression-target --init-token 0'
        )
    else:
        os.system(
            f'CUDA_VISIBLE_DEVICES=0 fairseq-train {TASK_PATH/"lipo"/tokenizer_suffix} --update-freq 1 --restore-file {MODEL_PATH/suffix/"checkpoint_last.pt"} --batch-size 1 --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --reset-optimizer --reset-dataloader --reset-meters --share-all-embeddings --share-decoder-input-output-embed --max-target-positions 1024 --required-batch-size-multiple 1 --arch {architecture} --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 0 --weight-decay 0.01 --clip-norm 0.1 --lr 0.0 --max-update 1 --warmup-updates 1 --keep-best-checkpoints 1 --num-classes 1 --save-dir {PREDICTION_MODEL_PATH/suffix} --best-checkpoint-metric loss --regression-target --init-token 0'
        )
    transplant_model(
        PREDICTION_MODEL_PATH/suffix/"checkpoint_last.pt",
        MODEL_PATH/suffix/"checkpoint_last.pt",
    )


def get_embeddings(model, dataset, source_dictionary, cuda=3):
    if isinstance(dataset, (Path, str)):
        dataset = list(load_indexed_dataset(str(dataset), source_dictionary))
    embeddings = []
    for sample in tqdm(dataset):
        sample = sample[:1020]
        if isinstance(model, BARTHubInterface):
            prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(
                device=f"cuda:{cuda}"
            )
            # same as in predict
            embedding = (
                model.model(
                    sample.unsqueeze(0).to(device=f"cuda:{cuda}"),
                    None,
                    prev_output_tokens,
                    features_only=True,
                )[0][0][-1, :]
                .cpu()
                .detach()
                .numpy()
            )
        else:
            embedding= model.model(sample.unsqueeze(0).to(device=f"cuda:{cuda}"), None)[0][0][-1, :].cpu().detach().numpy()
        embeddings.append(embedding)
    return embeddings


def generate_prev_output_tokens(sample: np.ndarray, source_dictionary) -> np.ndarray:
    """Generate previous output tokens needed for the fairseq models

    Args:
        sample (np.ndarray): sample to generate output tokens from
        source_dictionary (fairseq dictionary): used to translation

    Returns:
        np.ndarray: previous output tokens
    """
    tokens = sample.unsqueeze(0)
    prev_output_tokens = tokens.clone()
    prev_output_tokens[:, 0] = tokens.gather(
        1, (tokens.ne(source_dictionary.pad()).sum(1) - 1).unsqueeze(-1)
    ).squeeze()
    prev_output_tokens[:, 1:] = tokens[:, :-1]
    return prev_output_tokens


def compute_attention_output(#dataset: List[np.ndarray], 
    model, texts: List[str], source_dictionary, tokenizer=None
) -> List[List[Tuple[float, str]]]:
    """Compute attention of whole dataset with model copied from fairseq
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py

    Args:
        #dataset (List[np.ndarray]): pre-processed dataset to get attention from
        model (fairseq model): fairseq model
        texts (List[str]): human readable string of samples
        source_dictionary: source dictionary for fairseq
        tokenizer (optional): HuggingFace tokenizer to tokenize. Defaults to None.

    Returns:
        List[List[Tuple[float, str]]]: List[List[attention, token]]
    """
    # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py
    device = next(model.parameters()).device
    dataset_attentions = []
    for counter, text in enumerate(texts):
        if tokenizer is None:
            parsed_tokens = [
                parsed_token
                for parsed_token in re.split(PARSING_REGEX, text)
                if parsed_token
            ]
        else:
            parsed_tokens = tokenizer.convert_ids_to_tokens(
                tokenizer(text).input_ids
            )
        sample = torch.tensor(tokenizer(text).input_ids)
        prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(
            device
        )
        # same as in predict
        _, extra = model.model(sample.unsqueeze(0).to(device), None, prev_output_tokens, features_only=False, return_all_hiddens=False)
        token_attention = extra["attn"][0][0][-1].cpu().detach().tolist()
        dataset_attentions.append(list(zip(token_attention, parsed_tokens)))
        # Attention does not add up to 1, because [CLS] token at the end takes a lot of attention
    return dataset_attentions


def compute_embedding_output(
    dataset: List[np.ndarray], model, text: List[str], source_dictionary, tokenizer=None
) -> List[List[Tuple[float, str]]]:
    """Compute embedding of whole dataset with model copied from fairseq
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py

    Args:
        dataset (List[np.ndarray]): pre-processed dataset to get embeddings from
        model (fairseq model): fairseq model
        text (List[str]): human readable string of samples
        source_dictionary: source dictionary for fairseq
        tokenizer (optional): HuggingFace tokenizer to tokenize. Defaults to None.

    Returns:
        List[List[Tuple[float, str]]]: List[List[embedding, token]]
    """
    # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py
    device = next(model.parameters()).device
    dataset_embeddings = []
    for counter, sample in enumerate(dataset):
        if tokenizer is None:
            parsed_tokens = [
                parsed_token
                for parsed_token in re.split(PARSING_REGEX, text[counter])
                if parsed_token
            ]
        else:
            parsed_tokens = tokenizer.convert_ids_to_tokens(
                tokenizer(str(text[counter])).input_ids
            )
        prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(
            device
        )
        # same as in predict
        token_embeddings, _ = model.model(
            sample.unsqueeze(0).to(device), None, prev_output_tokens, features_only=True
        )
        dataset_embeddings.append(
            list(zip(token_embeddings[0][-1].cpu().detach().tolist(), parsed_tokens))
        )
    return dataset_embeddings


def compute_model_output(
    dataset: List[np.ndarray],
    model,
    text: List[str],
    source_dictionary,
    attentions=True,
    eos_attentions=True,
    embeddings=False,
    eos_embedding=False,
    tokenizer=None,
) -> Tuple[
    List[List[Tuple[float, str]]],
    List[List[Tuple[List[float], str]]],
    List[Tuple[List[float], str]],
]:
    """Compute embedding of whole dataset with model copied from fairseq
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py

    Args:
        dataset (List[np.ndarray]): pre-processed dataset to get embeddings from
        model (fairseq model): fairseq model
        text (List[str]): human readable string of samples
        source_dictionary: source dictionary for fairseq
        tokenizer (optional): HuggingFace tokenizer to tokenize. Defaults to None.

    Returns:
        List[List[Tuple[float, str]]]: List[List[embedding, token]]
    """
    # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/hub_interface.py
    device = next(model.parameters()).device
    dataset_attentions = [] if attentions else None
    dataset_eos_attentions = [] if eos_attentions else None
    dataset_embeddings = [] if embeddings else None
    dataset_eos_embeddings = [] if eos_embedding else None
    for counter, sample in enumerate(dataset):
        if tokenizer is None:
            parsed_tokens = [
                parsed_token
                for parsed_token in re.split(PARSING_REGEX, text[counter])
                if parsed_token
            ]
        else:
            parsed_tokens = tokenizer.convert_ids_to_tokens(
                tokenizer(str(text[counter])).input_ids
            )
        prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(
            device
        )
        # same as in predict
        token_embeddings, extra = model.model(
            sample.unsqueeze(0).to(device), None, prev_output_tokens, features_only=False
        )
        if attentions or eos_attentions:
            attention = extra["attn"][0][0].cpu().detach().tolist()
        if attentions:
            dataset_attentions.append(list(zip(attention, parsed_tokens)))
        if eos_attentions:
            dataset_eos_attentions.append(list(zip(attention[-1], parsed_tokens)))
        if embeddings or eos_embedding:
            token_embeddings = token_embeddings[0].cpu().detach().tolist()
        if embeddings:
            dataset_embeddings.append(list(zip(token_embeddings, parsed_tokens)))
        if eos_embedding:
            dataset_eos_embeddings.append(token_embeddings[-1])
    return (
        dataset_attentions,
        dataset_eos_attentions,
        dataset_embeddings,
        dataset_eos_embeddings,
    )


def preprocess_series(series: np.ndarray, output_path: Path, dictionary_path: Path):
    file_path = str(output_path) + ".csv"
    os.makedirs(output_path.parent, exist_ok=True)
    series.tofile(file_path, sep="\n", format="%s")
    os.system(
        (
            f"fairseq-preprocess --only-source --trainpref {file_path}  --destdir {output_path} --srcdict {dictionary_path} --workers 60"
        )
    )

def create_random_prediction_model(output_path:Path):
    if not output_path.exists():
        os.system(f'CUDA_VISIBLE_DEVICES=6 micromamba run -n fairseq_git fairseq-train {FAIRSEQ_PREPROCESS_PATH}/smiles_atom_isomers --save-dir {MODEL_PATH/"random_bart"} --max-source-positions 1024 --batch-size 32 --mask 0.0 --tokens-per-sample 512 --max-update 1 --warmup-updates 1 --task denoising --save-interval 1 --arch bart_base --optimizer adam --lr 0.0 --dropout 0.0 --criterion cross_entropy --max-tokens 3200 --weight-decay 0.0 --attention-dropout 0.0 --relu-dropout 0.0 --share-decoder-input-output-embed --share-all-embeddings --clip-norm 1.0 --skip-invalid-size-inputs-valid-test --log-format json --seed 4 --distributed-world-size 1 --no-epoch-checkpoints --mask-length span-poisson --encoder-learned-pos --decoder-learned-pos --rotate 0.0 --mask-random 0.0 --insert 0.0 --poisson-lambda 3.5 --dataset-impl mmap --num-workers 4')
        print("hey")
        os.system(f'CUDA_VISIBLE_DEVICES=6 fairseq-train {TASK_PATH/"lipo"/"smiles_atom_isomers"} --restore-file {MODEL_PATH/"random_bart"/"checkpoint_last.pt"} --save-dir {output_path} --max-source-positions 1024 --update-freq 1 --batch-size 1 --task sentence_prediction --num-workers 1 --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --required-batch-size-multiple 1 --add-prev-output-tokens --reset-optimizer --reset-dataloader --reset-meters --reset-lr-scheduler --arch bart_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 1024 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 0.1 --clip-norm 0.1 --lr 0.0 --max-update 2 --warmup-updates 1 --keep-best-checkpoints 1 --num-classes 1 --best-checkpoint-metric loss --regression-target --init-token 0')

def get_dictionary(target_dict_path):
    return Dictionary.load(str(target_dict_path))