"""Testing the interpretability of BART embeddings
SMILES or SELFIES, 2023
"""
import logging
import os
import pprint
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fairseq.data import Dictionary
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from constants import (
    DESCRIPTORS,
    FAIRSEQ_PREPROCESS_PATH,
    PROCESSED_PATH,
    PROJECT_PATH,
    SEED,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
)
from fairseq_utils import (
    get_embeddings,
    transform_to_translation_models,
    transplant_model,
)
from plotting import plot_representations
from preprocessing import get_weight
from scoring import load_dataset, load_model
from tokenisation import get_tokenizer, tokenize_dataset
from utils import parse_arguments, pickle_object

os.environ["MKL_THREADING_LAYER"] = "GNU"


def eval_weak_estimators(
    train_X: pd.DataFrame,
    train_y: np.array,
    test_X: pd.DataFrame,
    test_y: np.array,
    report_prefix: Path,
):
    """Train and evaluate the weaker sklearn classifiers on the training set and save the results to report_prefix

    Args:
        train_X (pd.DataFrame): training features
        train_y (np.array): training labels
        test_X (pd.DataFrame): testing features
        test_y (np.array): testing labels
        report_prefix (Path): report path
    """
    estimators = {
        "KNN": KNeighborsClassifier(),
        "RBF SVC": SVC(kernel="rbf", random_state=SEED + 49057),
        "Linear SVC": LinearSVC(random_state=SEED + 57, max_iter=100000),
        "Logistic Regression": LogisticRegression(
            random_state=SEED + 497, solver="saga", max_iter=10000
        ),
    }
    for name, estimator in estimators.items():
        if name == "KNN":
            param_grid = {"n_neighbors": [1, 5, 11], "weights": ["uniform", "distance"]}
        else:
            param_grid = {"C": [0.1, 1, 10]}  # coef0 for SVC?
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=16,
            verbose=4,
        )
        grid_search.fit(train_X, train_y)
        predictions = grid_search.predict(test_X)
        reportpath = report_prefix / f"estimator_{name}.txt"
        os.makedirs(reportpath.parent, exist_ok=True)
        with open(reportpath, "w") as reportfile:
            reportfile.writelines(
                [
                    str(name),
                    str(classification_report(test_y, predictions)),
                    pprint.pformat(grid_search.best_params_),
                ]
            )


def parse_column_name(
    column_name: Union[Tuple[str, List[str]], str]
) -> Tuple[str, List[str]]:
    """Properly parse the column name input to dataframe title and column names in source dataframe

    Args:
        column_name (Union[Tuple[str, List[str]], str]): input construct

    Returns:
        Tuple[str, List[str]]: title and column names in source dataframe
    """
    if isinstance(column_name, tuple):
        title = column_name[0]
        column_ids = column_name[1]
    else:
        title = column_name
        column_ids = [column_name]
    indexes = [*[str(DESCRIPTORS.index(column_id)) for column_id in column_ids], "210"]
    return title, indexes


def create_selected_dataframe(
    indexes: List[str], data_path: Path, amount: Optional[int] = None
) -> pd.DataFrame:
    """Create a dataframe with the columns indicated by indexes from data_path. With size of amount.

    Args:
        indexes (List[str]): Which indexes to read from file.
        data_path (Path): Which file to read
        amount (Optional[int], optional): Amount of size for each class. Defaults to biggest balanced dataset.

    Returns:
        pd.DataFrame: selected dataframe
    """
    df = pd.read_csv(data_path, skiprows=0, usecols=indexes)
    df = df.dropna()
    property_col = df[indexes[0]]
    for index in indexes[1:-1]:
        property_col += df[index]
    if amount is None:
        amount = min(len(df) - sum(property_col.gt(0)), sum(property_col.gt(0)))
    zeros = df[property_col == 0].sample(n=amount, random_state=SEED + 3497975)
    zeros["label"] = 0
    bigger = df[property_col > 0].sample(n=amount, random_state=SEED + 3497975)
    bigger["label"] = 1
    df = pd.concat([zeros, bigger]).drop(columns=indexes[:-1])
    return df


def create_dataset(
    column_name: Union[Tuple[str, List[str]], str],
    data_path: Path,
    output_path: Path,
    amount: Optional[int] = None,
):
    """Create a dataset with the columns specified in column_name from data_path with size amount and save it to output_path.

    Args:
        column_name (Union[Tuple[str, List[str]], str]): Specified columns
        data_path (Path): Which file to read
        output_path (Path): Where to save created dataset
        amount (Optional[int], optional): Amount of size for each class. Defaults to biggest balanced dataset.
    """
    title, indexes = parse_column_name(column_name)
    df = create_selected_dataframe(indexes, data_path, amount)
    train_SMILES, test_SMILES, train_y, test_y = train_test_split(
        df["210"],
        df["label"],
        test_size=0.2,
        random_state=SEED + 1233289,
        stratify=df["label"],
    )
    for tokenizer_suffix in TOKENIZER_SUFFIXES:
        logging.info(f"Creating embedding dataset for {tokenizer_suffix}")
        output_dir = output_path / title / tokenizer_suffix
        os.makedirs(output_dir, exist_ok=True)
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        selfies = tokenizer_suffix.startswith("selfies")
        train_mol = tokenize_dataset(tokenizer, train_SMILES, selfies)
        test_mol = tokenize_dataset(tokenizer, test_SMILES, selfies)
        train_mol.tofile(output_dir / "train.input", sep="\n", format="%s")
        test_mol.tofile(output_dir / "test.input", sep="\n", format="%s")
        train_y.to_numpy().tofile(output_dir / "train.label", sep="\n", format="%s")
        test_y.to_numpy().tofile(output_dir / "test.label", sep="\n", format="%s")
        # model_dict = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
        # os.system(
        #    (
        #        f'fairseq-preprocess --only-source --trainpref {output_dir/"train.input"} --testpref {output_dir/"test.input"} --destdir {output_dir/"input0"} --srcdict {model_dict} --workers 60'
        #    )
        # )


if __name__ == "__main__":
    cuda = parse_arguments(True, False, False)["cuda"]
    # transform_to_translation_models()
    #    for tokenizer_suffix in TOKENIZER_SUFFIXES:
    #        transplant_model(
    #            taker_model_path=PROJECT_PATH
    #            / "translation_models"
    #            / tokenizer_suffix
    #            / "checkpoint_last.pt",
    #            giver_model_path=PROJECT_PATH
    #            / "fairseq_models"
    #            / tokenizer_suffix
    #            / "checkpoint_last.pt",
    #        )
    #
    for descriptor in [
        (
            "Heterocycles",
            [
                "NumAliphaticHeterocycles",
                "NumAromaticHeterocycles",
                "NumSaturatedHeterocycles",
            ],
        ),
        # "NumHDonors",
        # "NumAromaticRings",
    ]:
        #        create_dataset(
        #            descriptor,
        #            PROCESSED_PATH / "10m_deduplicated.csv",
        #            PROJECT_PATH / "embeddings",
        #            100000,
        #        )
        if isinstance(descriptor, tuple):
            descriptor_name = descriptor[0]
        else:
            descriptor_name = descriptor
        weights = None
        for tokenizer_suffix in TOKENIZER_SUFFIXES:
            #            transplant_model(
            #                taker_model_path=PROJECT_PATH
            #                / "translation_models"
            #                / tokenizer_suffix
            #                / "checkpoint_last.pt",
            #                giver_model_path=PROJECT_PATH
            #                / "fairseq_models"
            #                / tokenizer_suffix
            #                / "checkpoint_last.pt",
            #            )
            model = load_model(
                PROJECT_PATH
                / "translation_models"
                / tokenizer_suffix
                / "checkpoint_last.pt",
                PROJECT_PATH / "embeddings" / tokenizer_suffix,
                str(cuda),
            )
            dataset_dict = {}
            descriptor_path = (
                PROJECT_PATH / "embeddings" / descriptor_name / tokenizer_suffix
            )
            embeddings_path = descriptor_path / "pickle"
            if weights is None:
                smiles = pd.read_csv(
                    descriptor_path / "train.input", header=None
                ).values
                weights = [
                    max(600, get_weight("".join(smile[0].split(" "))))
                    for smile in smiles
                ]

            for set_variation in ["train"]:  # , "test"]:
                dataset = load_dataset(descriptor_path / "input0" / set_variation)
                source_dictionary = Dictionary.load(
                    str(descriptor_path / "input0" / "dict.txt")
                )
                embeddings = get_embeddings(model, dataset, source_dictionary, cuda)
                plot_representations(
                    embeddings, weights, Path(f"plots/embeddings/{tokenizer_suffix}")
                )
#                os.makedirs(embeddings_path, exist_ok=True)
#                pickle_object(
#                    embeddings,
#                    descriptor_path / f"{tokenizer_suffix}_{set_variation}.pkl",
#                )
#                dataset_dict[f"{set_variation}_X"] = embeddings
#                dataset_dict[f"{set_variation}_y"] = np.fromfile(
#                    descriptor_path / f"{set_variation}.label", sep="\n"
#                )
#            eval_weak_estimators(
#                dataset_dict["train_X"],
#                dataset_dict["train_y"],
#                dataset_dict["test_X"],
#                dataset_dict["test_y"],
#                descriptor_path / "reports",
#            )
#
