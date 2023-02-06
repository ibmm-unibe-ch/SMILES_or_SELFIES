from pathlib import Path
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.bart import BARTModel
import json
import os
from constants import TASK_PATH, PROJECT_PATH


def load_model(model_path: Path, data_path: Path, cuda_device: str = None):
    """Load fairseq BART model

    Args:
        model_path (Path): path to .pt file
        cuda_device (str, optional): if model should be converted to a device. Defaults to None.

    Returns:
        fairseq_model: load BART model
    """
    model = BARTModel.from_pretrained(
        str(model_path.parent),
        data_name_or_path=str(data_path),
        checkpoint_file=str(model_path.name),
    )
    model.eval()
    if cuda_device:
        model.cuda(device=str(f"cuda:{cuda_device}"))
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


def transform_to_translation_models():
    for tokenizer_suffix in [
        "smiles_atom",
        "selfies_atom",
        "smiles_sentencepiece",
        "selfies_sentencepiece",
    ]:
        os.system(
            f'CUDA_VISIBLE_DEVICES=0 fairseq-train {TASK_PATH/"lipo"/tokenizer_suffix} --update-freq 1 --restore-file {PROJECT_PATH/"fairseq_models"/tokenizer_suffix/"checkpoint_last.pt"} --batch-size 1 --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 1024 --max-source-positions 1024 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 0 --weight-decay 0.01 --clip-norm 0.1 --lr 0.0 --max-update 1 --warmup-updates 1 --fp16 --keep-best-checkpoints 1 --num-classes 1 --save-dir {PROJECT_PATH/"translation_models"/tokenizer_suffix} --best-checkpoint-metric loss --regression-target --init-token 0'
        )
        transplant_model(
            PROJECT_PATH
            / "translation_models"
            / tokenizer_suffix
            / "checkpoint_last.pt",
            PROJECT_PATH / "fairseq_models" / tokenizer_suffix / "checkpoint_last.pt",
        )
