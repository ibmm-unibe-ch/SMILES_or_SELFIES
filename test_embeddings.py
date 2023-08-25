"""TBD
SMILES or SELFIES, 2023
"""
import logging
import os
import pprint

import numpy as np
import pandas as pd
from attention_readout import generate_prev_output_tokens
from constants import (
    DESCRIPTORS,
    FAIRSEQ_PREPROCESS_PATH,
    PROJECT_PATH,
    SEED,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
)
from scoring import load_dataset, load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from tokenisation import get_tokenizer, tokenize_dataset
from tqdm import tqdm
from utils import parse_arguments, pickle_object, unpickle

from fairseq.data import Dictionary

os.environ["MKL_THREADING_LAYER"] = "GNU"


def eval_weak_estimators(train_X, train_y, test_X, test_y, report_prefix):
    estimators = {
        "KNN": KNeighborsClassifier(),
        "RBF SVC": SVC(kernel="rbf", random_state=SEED + 49057),
        "Linear SVC": LinearSVC(random_state=SEED + 57),
        "Logistic Regression": LogisticRegression(random_state=SEED + 497),
    }
    for name, estimator in estimators.items():
        if name == "KNN":
            param_grid = {"n_neighbors": [11], "weights": ["distance"]}
        else:
            param_grid = {"C": [0.1, 1, 10]}  # coef0 for SVC?
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=3,
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


def create_dataset(column_name, data_path, output_path):
    indexes = [str(DESCRIPTORS.index(column_name)), "210"]
    df = pd.read_csv(data_path, skiprows=0, usecols=indexes)
    df = df.dropna()
    amount = min(len(df) - sum(df[indexes[0]].gt(0)), sum(df[indexes[0]].gt(0)))
    zeros = df[df[indexes[0]] == 0].sample(n=amount, random_state=SEED + 3497975)
    zeros["label"] = 0
    bigger = df[df[indexes[0]] > 0].sample(n=amount, random_state=SEED + 3497975)
    bigger["label"] = 1
    df = pd.concat([zeros, bigger]).drop(columns=[indexes[0]])
    train_SMILES, test_SMILES, train_y, test_y = train_test_split(
        df["210"],
        df["label"],
        test_size=0.2,
        random_state=SEED + 1233289,
        stratify=df["label"],
    )
    for tokenizer_suffix in TOKENIZER_SUFFIXES:
        logging.info(f"Creating embedding dataset for {tokenizer_suffix}")
        model_dict = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
        output_dir = output_path / tokenizer_suffix
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        selfies = tokenizer_suffix.startswith("selfies")
        train_mol = tokenize_dataset(tokenizer, train_SMILES, selfies)
        test_mol = tokenize_dataset(tokenizer, test_SMILES, selfies)
        os.makedirs(output_dir, exist_ok=True)
        train_mol.tofile(output_dir / "train.input", sep="\n", format="%s")
        test_mol.tofile(output_dir / "test.input", sep="\n", format="%s")
        train_y.to_numpy().tofile(output_dir / "train.label", sep="\n", format="%s")
        test_y.to_numpy().tofile(output_dir / "test.label", sep="\n", format="%s")
        os.system(
            (
                f'fairseq-preprocess --only-source --trainpref {output_dir/"train.input"} --testpref {output_dir/"test.input"} --destdir {output_dir/"input0"} --srcdict {model_dict} --workers 60'
            )
        )


def get_embeddings(model, dataset, source_dictionary, cuda=3):
    embeddings = []
    for sample in tqdm(dataset):
        sample = sample[:1020]
        prev_output_tokens = generate_prev_output_tokens(sample, source_dictionary).to(
            device=f"cuda:{cuda}"
        )
        # same as in predict
        features = model.model(
            sample.unsqueeze(0).to(device=f"cuda:{cuda}"),
            None,
            prev_output_tokens,
            features_only=True,
        )[0][-1, :]
        embedding = (
            features.view(features.size(0), -1, features.size(-1))[:, -1, :]
            .cpu()
            .detach()
            .numpy()
        ).squeeze()
        embeddings.append(embedding)
    return embeddings


def main():
    # create_dataset(
    #    "NumHDonors",
    #    PROCESSED_PATH / "10m_deduplicated.csv",
    #    PROJECT_PATH / "embeddings",
    # )
    # transform_to_translation_models()
    cuda = parse_arguments(True, False, False)["cuda"]
    flag = False
    for tokenizer_suffix in reversed(TOKENIZER_SUFFIXES):
        _ = """
        transplant_model(
            taker_model_path=PROJECT_PATH
            / "translation_models"
            / tokenizer_suffix
            / "checkpoint_last.pt",
            giver_model_path=PROJECT_PATH
            / "fairseq_models"
            / tokenizer_suffix
            / "checkpoint_last.pt",
        )
        """
        model = load_model(
            PROJECT_PATH
            / "translation_models"
            / tokenizer_suffix
            / "checkpoint_last.pt",
            PROJECT_PATH / "embeddings" / tokenizer_suffix,
            str(cuda),
        )
        dataset_dict = {}
        embeddings_path = PROJECT_PATH / "embeddings" / "pickle"
        for set_variation in ["train", "test"]:
            if flag:
                dataset = load_dataset(
                    PROJECT_PATH
                    / "embeddings"
                    / tokenizer_suffix
                    / "input0"
                    / set_variation
                )
                source_dictionary = Dictionary.load(
                    str(
                        PROJECT_PATH
                        / "embeddings"
                        / tokenizer_suffix
                        / "input0"
                        / "dict.txt"
                    )
                )

                embeddings = get_embeddings(model, dataset, source_dictionary, cuda)
                os.makedirs(embeddings_path, exist_ok=True)
                pickle_object(
                    embeddings,
                    embeddings_path / f"{tokenizer_suffix}_{set_variation}.pkl",
                )
            else:
                embeddings = unpickle(
                    embeddings_path / f"{tokenizer_suffix}_{set_variation}.pkl"
                )
            dataset_dict[f"{set_variation}_X"] = embeddings
            dataset_dict[f"{set_variation}_y"] = np.fromfile(
                PROJECT_PATH
                / "embeddings"
                / tokenizer_suffix
                / f"{set_variation}.label",
                sep="\n",
            )
        report_prefix = PROJECT_PATH / "embeddings" / tokenizer_suffix / "reports"
        eval_weak_estimators(
            dataset_dict["train_X"],
            dataset_dict["train_y"],
            dataset_dict["test_X"],
            dataset_dict["test_y"],
            report_prefix,
        )
        flag = True


if __name__ == "__main__":
    main()
