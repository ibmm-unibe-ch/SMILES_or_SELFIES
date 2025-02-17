"""Plotting of embeddings of selected molecules
SMILES or SELFIES, 2023
"""

from pathlib import Path
import random
from typing import Optional

import numpy as np
import pandas as pd
from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    PLOT_PATH,
    PROCESSED_PATH,
    PROJECT_PATH,
    SEED,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
    TASK_PATH,
    PREDICTION_MODEL_PATH,
)
from fairseq_utils import get_embeddings, load_model, preprocess_series, create_random_prediction_model, transform_to_prediction_model 
from plotting import plot_representations
from preprocessing import canonize_smile, check_valid, translate_selfie
from sample_molecules import BETA_LACTAMS, STEROIDS, TROPANES, SULFONAMIDES
from tokenisation import get_tokenizer, tokenize_to_ids
from utils import parse_arguments

from fairseq.data import Dictionary


def prepare_selected_molecules(sample_amount:Optional[int]=64):
    random.seed(4)
    iterator_variables = zip(
        [random.sample(list(set(STEROIDS)),sample_amount), random.sample(list(set(BETA_LACTAMS)),sample_amount), random.sample(list(set(TROPANES)),sample_amount), random.sample(list(set(SULFONAMIDES)),sample_amount)],
        ["Steroids", "Beta lactams", "Tropanes", "Sulfonamides"],
    )
    output = []
    output_labels = []
    for dataset, name in iterator_variables:
        dataset = set(dataset)
        dataset = [
            (canonize_smile(smile), translate_selfie(smile)[0])
            for smile in dataset
            if (check_valid(smile) and translate_selfie(smile)[1] > 0)
        ]
        labels = [name] * len(dataset)
        output.extend(dataset)
        output_labels.extend(labels)
    output_smiles = [element[0] for element in output]
    output_selfies = [element[1] for element in output]
    return pd.DataFrame(
        {"SELFIES": output_selfies, "SMILES": output_smiles, "label": output_labels}
    )


def sample_other_molecules(data_path: Path, amount: int):
    if amount<=0:
        return None
    df = pd.read_csv(data_path, skiprows=0, usecols=["SMILES", "SELFIES"])
    df = df.dropna()
    df = df.sample(n=amount, random_state=SEED + 39775)
    df["label"] = "Other molecule"
    return df


def get_molecule_dataframe(
    data_path: Path,
):
    selected_df = prepare_selected_molecules(64)
    other_df = sample_other_molecules(data_path, amount=0)#selected_df.shape[0])
    df = pd.concat([selected_df, other_df], axis="index", ignore_index=True)
    return df


if __name__ == "__main__":
    variables = parse_arguments(True, True, False, False, False, True,)
    cuda = variables["cuda"]
    if "tokenizer" in variables and variables["tokenizer"] is not None:
        tokenizer_suffixes = [variables["tokenizer"]]
    else:
        tokenizer_suffixes = TOKENIZER_SUFFIXES
    modeltype = variables["modeltype"]
    molecule_dataframe = get_molecule_dataframe(
        PROCESSED_PATH/ "standard" / "full_deduplicated_standard.csv"
    )
    if modeltype == "random":
        tokenizer_suffix = "smiles_atom_isomers"
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        molecule_series = molecule_dataframe["SMILES"]
        tokenized_molecules = tokenize_to_ids(tokenizer, molecule_series)
        fairseq_dict_path = FAIRSEQ_PREPROCESS_PATH/tokenizer_suffix/"dict.txt"
        data_path = TASK_PATH/"bbbp"/tokenizer_suffix
        preprocess_series(tokenized_molecules, Path(PROJECT_PATH / f"embedding_mapping_{modeltype}"), fairseq_dict_path)
        model_path = PREDICTION_MODEL_PATH/"random_bart"/"checkpoint_last.pt"
        if not model_path.exists():
            create_random_prediction_model(model_path.parent)
        model = load_model(model_path,data_path,str(cuda))
        fairseq_dict = Dictionary.load(str(fairseq_dict_path))
        mol_dataset_path = PROJECT_PATH / f"embedding_mapping_{modeltype}" / "train"
        embeddings = get_embeddings(model, mol_dataset_path, fairseq_dict, cuda)
        for min_dist in [0.5]:
            for n_neighbors in [15]:
                plot_representations(embeddings,molecule_dataframe["label"],PLOT_PATH / "selected_molecules" / "random",min_dist,n_neighbors, alpha=0.6, offset=3)
    else:
        for it, tokenizer_suffix in enumerate(tokenizer_suffixes):
            tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
            if tokenizer_suffix.startswith("selfies"):
                molecule_series = molecule_dataframe["SELFIES"]
            else:
                molecule_series = molecule_dataframe["SMILES"]
            tokenized_molecules = tokenize_to_ids(tokenizer, molecule_series)
            fairseq_dict_path = FAIRSEQ_PREPROCESS_PATH/tokenizer_suffix/"dict.txt"
            data_path = TASK_PATH/"bbbp"/tokenizer_suffix
            preprocess_series(tokenized_molecules, Path(PROJECT_PATH / f"embedding_mapping_{modeltype}"), fairseq_dict_path)
            tokenizer_model_suffix = tokenizer_suffix+"_"+modeltype
            model_path = PREDICTION_MODEL_PATH/tokenizer_model_suffix/"checkpoint_last.pt"
            if not model_path.exists():
                transform_to_prediction_model(tokenizer_model_suffix)
            model = load_model(model_path,data_path,str(cuda))
            fairseq_dict = Dictionary.load(str(fairseq_dict_path))
            mol_dataset_path = PROJECT_PATH / f"embedding_mapping_{modeltype}" / "train"
            embeddings = get_embeddings(model, mol_dataset_path, fairseq_dict, cuda)
            for min_dist in [0.5]:
                for n_neighbors in [15]:
                    plot_representations(embeddings,molecule_dataframe["label"],PLOT_PATH / "selected_molecules" / tokenizer_model_suffix,min_dist,n_neighbors, alpha=0.6,offset=it)
    