"""Testing the interpretability of BART embeddings for SMILES or SELFIES representations.

This module provides functionality for evaluating molecular embeddings using various
machine learning classifiers and regressors. It supports different tasks including
pretraining on molecular descriptors, MoleculeNet benchmarks, and ETH datasets.
"""

import logging
import os
import pprint
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    PROJECT_PATH,
    SEED,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
    PREDICTION_MODEL_PATH,
    TASK_PATH,
    MOLNET_DIRECTORY,
    PROCESSED_PATH
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from fairseq_utils import (
    get_embeddings,
    transform_to_prediction_model,
    load_dataset,
    load_model,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from tokenisation import get_tokenizer, tokenize_dataset
from utils import parse_arguments, pickle_object, unpickle
from fairseq.data import Dictionary

os.environ["MKL_THREADING_LAYER"] = "GNU"


def eval_weak_classifiers(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    report_prefix: Path,
    val_X: Optional[pd.DataFrame] = None,
    val_y: Optional[pd.DataFrame] = None,
    groups: Optional[pd.Series] = None,
) -> None:
    """Train and evaluate sklearn classifiers on the training set and save results.

    Args:
        train_X: Training features as a DataFrame.
        train_y: Training labels as a numpy array.
        test_X: Testing features as a DataFrame.
        test_y: Testing labels as a numpy array.
        report_prefix: Directory path to save evaluation reports.
        val_X: Optional validation features as a DataFrame (for MoleculeNet).
        val_y: Optional validation labels as a DataFrame (for MoleculeNet).
    """
    estimators = {
        "RBF SVC": SVC(kernel=rbf_kernel, random_state=SEED + 49057, cache_size=100, class_weight="balanced"),
        "KNN": KNeighborsClassifier(),
        "Linear SVC": LinearSVC(random_state=SEED + 57, max_iter=1000, class_weight="balanced"),
    }
    
    for name, estimator in estimators.items():
        param_grid = {"n_neighbors": [1, 5, 11], "weights": ["uniform", "distance"]} if name == "KNN" else {"C": [0.1, 1, 10]}
        
        if val_X is None:
            # Combine train and test for cross-validation
            if not (test_X is None):
                X = np.concatenate([train_X, test_X])
                y = np.concatenate([np.array(train_y), np.array(test_y)])
            else:
                X = np.array(train_X)
                y = np.array(train_y)            
            if not (groups is None):
                sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                splits = list(sgkf.split(X,y,groups=groups))
                grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="accuracy",
                cv=splits,
                n_jobs=10,
                verbose=4,
            )
            else:
                grid_search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    scoring="accuracy",
                    cv=3,
                    n_jobs=10,
                    verbose=4,
                )
        else:  # MoleculeNet case with validation set
            test_fold = [0] * len(train_X) + [1] * len(val_X)
            X = np.concatenate([train_X, val_X])
            y = np.concatenate([train_y, val_y])
            ps = PredefinedSplit(test_fold=test_fold)
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=ps,
                n_jobs=10,
                verbose=4,
            )
        
        grid_search.fit(X, y)
        lines = [f"{name}\n", pprint.pformat(grid_search.cv_results_) + "\n"]
        
        if val_X is not None:  # MoleculeNet case - add test results
            predictions = grid_search.predict(test_X)
            lines.extend([
                "CLASSIFICATION_TEST_RESULTS!!\n",
                str(roc_auc_score(test_y, predictions)) + "\n",
                str(classification_report(test_y, predictions)) + "\n",
            ])
        
        report_path = report_prefix / f"estimator_{name}.txt"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, "w") as report_file:
            report_file.writelines(lines)


def eval_weak_regressors(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    report_prefix: Path,
    val_X: Optional[pd.DataFrame] = None,
    val_y: Optional[pd.DataFrame] = None,
    groups: Optional[pd.Series] =None,
) -> None:
    """Train and evaluate sklearn regressors on the training set and save results.

    Args:
        train_X: Training features as a DataFrame.
        train_y: Training labels as a numpy array.
        test_X: Testing features as a DataFrame.
        test_y: Testing labels as a numpy array.
        report_prefix: Directory path to save evaluation reports.
        val_X: Optional validation features as a DataFrame (for MoleculeNet).
        val_y: Optional validation labels as a DataFrame (for MoleculeNet).
    """
    estimators = {
        "RBF SVR": SVR(kernel=rbf_kernel, cache_size=100),
        "KNN": KNeighborsRegressor(),
        "Linear SVR": LinearSVR(random_state=SEED + 57, max_iter=1000),
    }
    
    for name, estimator in estimators.items():
        param_grid = {"n_neighbors": [1, 5, 11], "weights": ["uniform", "distance"]} if name == "KNN" else {"C": [0.1, 1, 10]}
        if val_X is None:
            # Combine train and test for cross-validation
            if not (test_X is None):
                X = np.concatenate([train_X, test_X])
                y = np.concatenate([train_y, test_y])
            else:
                X = np.array(train_X)
                y = np.array(train_y)  
            if not (groups is None):
                sgkf = GroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                splits = list(sgkf.split(X,y,groups=groups))
                grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=splits,
                n_jobs=10,
                verbose=4,
            )
            else:
                grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=3,
                n_jobs=10,
                verbose=4,
            )
        else:  # MoleculeNet case with validation set
            test_fold = [0] * len(train_X) + [1] * len(val_X)
            X = np.concatenate([train_X, val_X])
            y = np.concatenate([train_y, val_y])
            ps = PredefinedSplit(test_fold=test_fold)
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=ps,
                n_jobs=10,
                verbose=4,
            )
        
        grid_search.fit(X, y)
        lines = [f"{name}\n", pprint.pformat(grid_search.cv_results_) + "\n"]
        
        if val_X is not None:  # MoleculeNet case - add test results
            predictions = grid_search.predict(test_X)
            rmse = root_mean_squared_error(test_y, predictions)
            mae = mean_absolute_error(test_y, predictions)
            r2 = r2_score(test_y, predictions)
            lines.extend([
                "REGRESSION_TEST_RESULTS!!\n",
                f"{rmse}\n",
                f"RMSE: {rmse} MAE: {mae} R2 score: {r2}\n"
            ])
        
        report_path = report_prefix / f"estimator_{name}.txt"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, "w") as report_file:
            report_file.writelines(lines)


def parse_column_name(
    column_name: Union[Tuple[str, List[str]], str]
) -> Tuple[str, List[str]]:
    """Parse column name input into title and column names in source dataframe.

    Args:
        column_name: Either a string (single column) or tuple of (title, columns).

    Returns:
        Tuple of (title, column_names) where column_names includes "SMILES".
    """
    if isinstance(column_name, tuple):
        title = column_name[0]
        column_ids = column_name[1] + ["SMILES"]
    else:
        title = column_name
        column_ids = [column_name, "SMILES"]
    return title, column_ids


def create_selected_dataframe(
    indexes: List[str],
    data_path: Path,
    amount: Optional[int] = None,
    max_len: int = 512,
    classification: bool = True,
) -> pd.DataFrame:
    """Create a balanced dataframe with specified columns from the data file.

    Args:
        indexes: Column names to read from file.
        data_path: Path to the data file.
        amount: Number of samples per class (None for maximum balanced size).
        max_len: Maximum SMILES string length to include.
        classification: Whether to create classification labels (else regression).

    Returns:
        DataFrame with selected columns and labels.
    """
    df = pd.read_csv(data_path, skiprows=0, usecols=indexes).dropna()
    df = df[df.SMILES.str.len() < max_len]
    property_col = df[indexes[0]]
    
    # Sum additional properties if multiple columns specified
    for index in indexes[1:-1]:  # exclude SMILES at -1
        property_col += df[index]
    
    if classification:
        if amount is None:
            amount = min(len(df) - sum(property_col.gt(0)), sum(property_col.gt(0)))
        zeros = df[property_col == 0].sample(n=int(amount), random_state=SEED + 3497975)
        zeros["label"] = 0
        bigger = df[property_col > 0].sample(n=int(amount), random_state=SEED + 3497975)
        bigger["label"] = 1
        return pd.concat([zeros, bigger]).drop(columns=indexes[:-1])
    else:
        if amount is None:
            amount = 100000
        df["label"] = property_col
        return df.sample(n=int(amount), random_state=SEED + 3497975)


def create_dataset(
    column_name: Union[Tuple[str, List[str]], str],
    data_path: Path,
    output_path: Path,
    amount: Optional[int] = None,
    classification: bool = True,
) -> None:
    """Create and save a dataset from specified columns in the data file.

    Args:
        column_name: Either a string (single column) or tuple of (title, columns).
        data_path: Path to the source data file.
        output_path: Directory to save the created dataset.
        amount: Number of samples per class (None for maximum balanced size).
        classification: Whether to create classification labels (else regression).
    """
    title, indexes = parse_column_name(column_name)
    if (output_path / title / "smiles_atom_isomers" / "input0" / "dict.txt").exists():
        return
    
    df = create_selected_dataframe(indexes, data_path, amount, classification=classification)
    train_SMILES, test_SMILES, train_y, test_y = train_test_split(
        df["SMILES"],
        df["label"],
        test_size=0.2,
        random_state=SEED + 1233289,
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
        
        model_dict = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
        dest_dir = output_dir / "input0"
        
        os.system(
            f'fairseq-preprocess --only-source --trainpref {output_dir/"train.input"} '
            f'--testpref {output_dir/"test.input"} --destdir {dest_dir} '
            f'--srcdict {model_dict} --workers 60'
        )


def create_atom_dataset(
    data_path: Path,
    output_path: Path,
    amount: Optional[int] = None,
) -> None:
    """Create a dataset for atom-level properties from the data file.

    Args:
        data_path: Path to the source data file.
        output_path: Directory to save the created dataset.
        amount: Number of samples to include (None for default size).
    """
    df = create_selected_dataframe(["SMILES"], data_path, amount, classification=False).drop_duplicates()
    train_SMILES, test_SMILES, train_y, test_y = train_test_split(
        df["SMILES"],
        df["label"],
        test_size=0.2,
        random_state=SEED + 1233289,
    )
    
    for tokenizer_suffix in TOKENIZER_SUFFIXES:
        logging.info(f"Creating embedding dataset for {tokenizer_suffix}")
        output_dir = output_path / "eth" / tokenizer_suffix
        os.makedirs(output_dir, exist_ok=True)
        
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        selfies = tokenizer_suffix.startswith("selfies")
        
        train_mol = tokenize_dataset(tokenizer, train_SMILES, selfies)
        test_mol = tokenize_dataset(tokenizer, test_SMILES, selfies)
        
        train_mol.tofile(output_dir / "train.input", sep="\n", format="%s")
        test_mol.tofile(output_dir / "test.input", sep="\n", format="%s")
        
        model_dict = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
        dest_dir = output_dir / "input0"
        
        os.system(
            f'fairseq-preprocess --only-source --trainpref {output_dir/"train.input"} '
            f'--testpref {output_dir/"test.input"} --destdir {dest_dir} '
            f'--srcdict {model_dict} --workers 60'
        )


def main_molnet(tokenizers: List[str], model_type: str, cuda: bool) -> None:
    """Run evaluation on MoleculeNet benchmark tasks.

    Args:
        tokenizers: List of tokenizer suffixes to evaluate.
        model_type: Type of model being evaluated.
        cuda: Whether to use CUDA for model inference.
    """
    for task_name in MOLNET_DIRECTORY.keys():
        for tokenizer_suffix in tokenizers:
            model_suffix = f"{tokenizer_suffix}_{model_type}"
            fairseq_dict_path = TASK_PATH / "bbbp" / tokenizer_suffix
            model_path = PREDICTION_MODEL_PATH / model_suffix / "checkpoint_last.pt"
            
            if not model_path.exists():
                transform_to_prediction_model(model_suffix)
            
            model = load_model(model_path, fairseq_dict_path, str(cuda))
            dataset_dict: Dict[str, Union[np.ndarray, pd.DataFrame]] = {}
            task_path = TASK_PATH / task_name / tokenizer_suffix
            descriptor_path = PROJECT_PATH / f"embeddings_{model_type}" / task_name / tokenizer_suffix
            
            for set_variation in ["train", "test", "valid"]:
                dataset = load_dataset(task_path / "input0" / set_variation)
                source_dictionary = Dictionary.load(str(task_path / "input0" / "dict.txt"))
                pickle_dir = descriptor_path / f"{set_variation}_embeddings" / "embeddings.pkl"
                
                if pickle_dir.exists():
                    embeddings = unpickle(pickle_dir)
                else:
                    embeddings = get_embeddings(
                        model, dataset, source_dictionary, whole_mol=True, cuda=cuda
                    )
                    pickle_object(embeddings, pickle_dir)
                
                dataset_dict[f"{set_variation}_X"] = embeddings
                dataset_dict[f"{set_variation}_y"] = np.fromfile(
                    task_path / f"{set_variation}.label", sep="\n"
                )
            
            if MOLNET_DIRECTORY[task_name]["dataset_type"] == "classification":
                eval_weak_classifiers(
                    dataset_dict["train_X"],
                    dataset_dict["train_y"],
                    dataset_dict["test_X"],
                    dataset_dict["test_y"],
                    task_path / "reports"/ model_type,
                    val_X=dataset_dict["valid_X"],
                    val_y=dataset_dict["valid_y"],
                )
            else:
                eval_weak_regressors(
                    dataset_dict["train_X"],
                    dataset_dict["train_y"],
                    dataset_dict["test_X"],
                    dataset_dict["test_y"],
                    task_path / "reports"/ model_type,
                    val_X=dataset_dict["valid_X"],
                    val_y=dataset_dict["valid_y"],
                )


def main_pretraining(tokenizers: List[str], model_type: str, cuda: bool) -> None:
    """Run evaluation on pretraining molecular descriptors.

    Args:
        tokenizers: List of tokenizer suffixes to evaluate.
        model_type: Type of model being evaluated.
        cuda: Whether to use CUDA for model inference.
    """
    descriptor_configs = [
        (False, "Chi0v"),
        (False, "Kappa1"),
        (False, "MolLogP"),
        (False, "MolMR"),
        (False, "QED"),
        (True, "NumHDonors"),
        (True, ("Heterocycles", [
            "NumAliphaticHeterocycles",
            "NumAromaticHeterocycles",
            "NumSaturatedHeterocycles",
        ])),
    ]
    for classification, descriptor in descriptor_configs:
        descriptor_name = descriptor[0] if isinstance(descriptor, tuple) else descriptor
        if not (PROJECT_PATH / f"embeddings_{model_type}" / descriptor_name).exists():
            if classification:
                amount = 50000
            else:
                amount = 100000
            create_dataset(
                descriptor,
                PROCESSED_PATH/"descriptors/merged.csv",
                PROJECT_PATH / f"embeddings_{model_type}",
                amount,
                classification=classification,
            )
        
        for tokenizer_suffix in tokenizers:
            model_suffix = f"{tokenizer_suffix}_{model_type}"
            fairseq_dict_path = TASK_PATH / "bbbp" / tokenizer_suffix
            model_path = PREDICTION_MODEL_PATH / model_suffix / "checkpoint_last.pt"
            
            if not model_path.exists():
                transform_to_prediction_model(model_suffix)
            
            model = load_model(model_path, fairseq_dict_path, str(cuda))
            descriptor_path = PROJECT_PATH / f"embeddings_{model_type}" / descriptor_name / tokenizer_suffix
            dataset_dict: Dict[str, Union[np.ndarray, pd.DataFrame]] = {}
            
            for set_variation in ["train", "test"]:
                dataset = load_dataset(descriptor_path / "input0" / set_variation)
                source_dictionary = Dictionary.load(str(descriptor_path / "input0" / "dict.txt"))
                pickle_dir = descriptor_path / "input0" / f"{set_variation}_embeddings" / "embeddings.pkl"
                
                if pickle_dir.exists():
                    embeddings = unpickle(pickle_dir)
                else:
                    embeddings = get_embeddings(
                        model, dataset, source_dictionary, whole_mol=True, cuda=cuda
                    )
                    pickle_object(embeddings, pickle_dir)
                
                dataset_dict[f"{set_variation}_X"] = embeddings
                dataset_dict[f"{set_variation}_y"] = np.fromfile(
                    descriptor_path / f"{set_variation}.label", sep="\n"
                )
            
            if classification:
                eval_weak_classifiers(
                    dataset_dict["train_X"],
                    dataset_dict["train_y"],
                    dataset_dict["test_X"],
                    dataset_dict["test_y"],
                    descriptor_path / "reports",
                )
            else:
                eval_weak_regressors(
                    dataset_dict["train_X"],
                    dataset_dict["train_y"],
                    dataset_dict["test_X"],
                    dataset_dict["test_y"],
                    descriptor_path / "reports",
                )

if __name__ == "__main__":
    args = parse_arguments(cuda=True, tokenizer=True, task=True, model_type=True)
    cuda = args["cuda"]
    tokenizers = [args["tokenizer"]] if args.get("tokenizer", None) else TOKENIZER_SUFFIXES
    model_type = args["modeltype"]
    
    if args["task"] == "pretraining":
        main_pretraining(tokenizers, model_type, cuda)
    elif args["task"] == "molnet":
        main_molnet(tokenizers, model_type, cuda)
    else:
        main_eth(tokenizers, model_type, cuda)