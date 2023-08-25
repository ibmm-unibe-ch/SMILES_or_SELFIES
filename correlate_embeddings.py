import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import selfies
from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    PROCESSED_PATH,
    PROJECT_PATH,
    SEED,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
)
from fairseq_utils import get_embeddings, preprocess_series
from plotting import plot_correlation
from preprocessing import canonize_smile
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist, cosine
from scipy.stats import pearsonr, spearmanr
from scoring import load_model
from tokenisation import get_tokenizer
from tqdm import tqdm
from utils import parse_arguments

from fairseq.data import Dictionary


def sample_synonym(
    seed: int = None, min_size: int = 5, max_size: int = 30
) -> Tuple[str, str, str, str]:
    random.seed(seed)
    found_fitting = False
    alphabet = (
        selfies.get_semantic_robust_alphabet()
    )  # Gets the alphabet of robust symbols
    while not found_fitting:
        size = random.randint(min_size, max_size)
        rnd_selfies = "".join(random.sample(list(alphabet), size))
        rnd_smiles = selfies.decoder(rnd_selfies)
        rnd_smiles = rnd_smiles.replace("-1", "-").replace("+1", "+")
        canon_smiles = canonize_smile(rnd_smiles)
        rnd_selfies = selfies.encoder(rnd_smiles)
        canon_selfies = selfies.encoder(canon_smiles)
        if (canon_smiles != rnd_smiles) and (canon_selfies != rnd_selfies):
            found_fitting = True
    return rnd_smiles, canon_smiles, rnd_selfies, canon_selfies


def sample_random_molecules(
    amount: int = 1000, overcompensation_factor: int = 1.1
) -> pd.DataFrame:
    molecule_list = []
    for seed in range(SEED, int(SEED + amount * overcompensation_factor)):
        molecule_list.append(sample_synonym(seed))
    df = pd.DataFrame(
        molecule_list,
        columns=["rnd_smiles", "canon_smiles", "rnd_selfies", "canon_selfies"],
    )
    df.drop_duplicates(["canon_smiles"], inplace=True)
    if df.shape[0] < amount:
        logging.info(
            f"Overcompensation factor of {overcompensation_factor} was not enough."
        )
        return sample_random_molecules(amount, overcompensation_factor * 1.1)
    return df.iloc[:amount]


def compute_distances(start: pd.Series, end: pd.Series) -> Tuple[float, float]:
    euclid = cdist([start], [end], "euclid")
    manhattan = cdist([start], [end], "cityblock")
    cos = cosine(start, end)
    return euclid[0][0], manhattan[0][0], cos


def compute_stats(distances: pd.Series, prefix: str) -> Dict[str, float]:
    result = {}
    result[f"{prefix}_avg"] = distances.mean()
    result[f"{prefix}_med"] = distances.median()
    result[f"{prefix}_std"] = distances.std()
    return result


def get_distances(
    tokenizer_suffix: str, start_mols_path: Path, end_mols_path: Path, cuda: str = "3"
):
    model = load_model(
        PROJECT_PATH / "translation_models" / tokenizer_suffix / "checkpoint_last.pt",
        PROJECT_PATH / "embeddings" / tokenizer_suffix,
        str(cuda),
    )
    source_dictionary = Dictionary.load(
        str(FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt")
    )
    start_embeddings = get_embeddings(
        model, start_mols_path / "train", source_dictionary, cuda=str(cuda)
    )
    end_embeddings = get_embeddings(
        model, end_mols_path / "train", source_dictionary, cuda=str(cuda)
    )
    euclideans = []
    manhattans = []
    cosines = []
    fixed_euclideans = []
    fixed_manhattans = []
    fixed_cosines = []
    for it, start_embedding in enumerate(start_embeddings):
        euclidean, manhattan, cos = compute_distances(
            end_embeddings[it], start_embedding
        )
        euclideans.append(euclidean)
        manhattans.append(manhattan)
        cosines.append(cos)
        fixed_euclidean, fixed_manhattan, fixed_cos = compute_distances(
            end_embeddings[SEED % len(end_embeddings)], start_embedding
        )
        fixed_euclideans.append(fixed_euclidean)
        fixed_manhattans.append(fixed_manhattan)
        fixed_cosines.append(fixed_cos)
    return (
        pd.Series(manhattans),
        pd.Series(euclideans),
        pd.Series(cosines),
        pd.Series(fixed_manhattans),
        pd.Series(fixed_euclideans),
        pd.Series(fixed_cosines),
    )


def tokenize_sample_to_ids(tokenizer, sample):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(str(sample)).input_ids)
    return " ".join(tokens)


def tokenize_to_ids(tokenizer, samples):
    return np.array(
        [tokenize_sample_to_ids(tokenizer, sample) for sample in tqdm(samples)]
    )


def preprocess_series_fairseq(
    folder_name: str, tokenizer_suffix: str, df: pd.DataFrame
):
    dictionary_path = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
    tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
    if tokenizer_suffix.startswith("selfies"):
        preprocess_series(
            tokenize_to_ids(tokenizer, df["start_selfies"]),
            PROJECT_PATH / folder_name / tokenizer_suffix / "start",
            dictionary_path,
        )
        preprocess_series(
            tokenize_to_ids(tokenizer, df["end_selfies"]),
            PROJECT_PATH / folder_name / tokenizer_suffix / "end",
            dictionary_path,
        )
    else:
        preprocess_series(
            tokenize_to_ids(tokenizer, df["start_smiles"]),
            PROJECT_PATH / folder_name / tokenizer_suffix / "start",
            dictionary_path,
        )
        preprocess_series(
            tokenize_to_ids(tokenizer, df["end_smiles"]),
            PROJECT_PATH / folder_name / tokenizer_suffix / "end",
            dictionary_path,
        )


def measure_synonyms(cuda: str):
    df = sample_random_molecules().rename(
        {
            "rnd_smiles": "start_smiles",
            "canon_smiles": "end_smiles",
            "rnd_selfies": "start_selfies",
            "canon_selfies": "end_selfies",
        },
        axis="columns",
    )
    dataframes = []
    for tokenizer_suffix in TOKENIZER_SUFFIXES:
        preprocess_series_fairseq("synonyms", tokenizer_suffix, df)
        manhattans, euclideans, cosines, _, _, _ = get_distances(
            tokenizer_suffix,
            PROJECT_PATH / "synonyms" / tokenizer_suffix / "start",
            PROJECT_PATH / "synonyms" / tokenizer_suffix / "end",
            cuda,
        )
        row = (
            compute_stats(manhattans, "manhattan")
            | compute_stats(euclideans, "euclidean")
            | compute_stats(cosines, "cosine")
        )
        dataframe = pd.DataFrame(data=row, index=[tokenizer_suffix])
        dataframes.append(dataframe)
    return pd.concat(dataframes)


def get_pairwise_chemical_similarities(mol_start, mol_end):
    morgan_start = AllChem.GetMorganFingerprint(mol_start, 2)  # 2 is closest to ECFP
    morgan_end = AllChem.GetMorganFingerprint(mol_end, 2)  # 2 is closest to ECFP
    morgan_dice = DataStructs.DiceSimilarity(morgan_start, morgan_end)
    morgan_start = AllChem.GetMorganFingerprintAsBitVect(
        mol_start, 2
    )  # count is needed for Tanimoto
    morgan_end = AllChem.GetMorganFingerprintAsBitVect(
        mol_end, 2
    )  # count is needed for Tanimoto
    morgan_tanimoto = DataStructs.FingerprintSimilarity(morgan_start, morgan_end)
    rdkit_start = Chem.RDKFingerprint(mol_start)
    rdkit_end = Chem.RDKFingerprint(mol_end)
    rdkit_distance = DataStructs.FingerprintSimilarity(rdkit_start, rdkit_end)
    return morgan_dice, morgan_tanimoto, rdkit_distance


def get_chemical_similarities(SMILES_starts, SMILES_ends):
    mol_starts = [Chem.MolFromSmiles(SMILES_start) for SMILES_start in SMILES_starts]
    mol_ends = [Chem.MolFromSmiles(SMILES_end) for SMILES_end in SMILES_ends]
    similarities = np.concatenate(
        [
            np.array(
                [
                    get_pairwise_chemical_similarities(mol_start, mol_end)
                    for (mol_start, mol_end) in zip(mol_starts, mol_ends)
                ]
            ),
            np.array(
                [
                    get_pairwise_chemical_similarities(
                        mol_start, mol_ends[SEED % len(mol_ends)]
                    )
                    for mol_start in mol_starts
                ]
            ),
        ],
        axis=1,
    )
    return similarities


def select_molecules_from_df(dataframe_path: Path, amount: int = 1000, seed: int = 0):
    dataframe = pd.read_csv(dataframe_path, usecols=["210", "208"])
    selected_df = dataframe.sample(n=amount, random_state=seed + SEED)
    selected_df.rename(
        {"210": "smiles", "208": "selfies"}, axis="columns", inplace=True
    )
    return selected_df


def get_correlation(distances1, distances2, prefix: Optional[str] = None):
    pearson = pearsonr(distances1, distances2)
    spearman = spearmanr(distances1, distances2)
    pearson_string = f"{prefix} Pearson" if prefix else "Pearson"
    spearman_string = f"{prefix} Spearman" if prefix else "Spearman"
    return {pearson_string: pearson, spearman_string: spearman}


def find_correlations_different_starts(
    dataframe_path: Path, amount: int = 1000, cuda: str = "3"
):
    starts_df = select_molecules_from_df(dataframe_path, amount, seed=0)
    ends_df = select_molecules_from_df(dataframe_path, amount, seed=1)
    logging.info("Sampled starts and ends for correlation experiment.")
    chemical_similarities = get_chemical_similarities(
        starts_df["smiles"], ends_df["smiles"]
    )
    morgan_dice = chemical_similarities[:, 0]
    morgan_tanimoto = chemical_similarities[:, 1]
    rdkit_distance = chemical_similarities[:, 2]
    fixed_morgan_dice = chemical_similarities[:, 3]
    fixed_morgan_tanimoto = chemical_similarities[:, 4]
    fixed_rdkit = chemical_similarities[:, 5]
    logging.info(
        "Computed Morgan distances for starts and ends of correlation computation."
    )
    scores = []
    plotting_distances = {}
    plotting_distances_fixed = {}
    for tokenizer_suffix in TOKENIZER_SUFFIXES:
        if tokenizer_suffix.startswith("selfies"):
            df = pd.concat(
                [
                    starts_df.drop(["smiles"], axis="columns")
                    .rename({"selfies": "start_selfies"}, axis="columns")
                    .reset_index(drop=True),
                    ends_df.drop(["smiles"], axis="columns")
                    .rename({"selfies": "end_selfies"}, axis="columns")
                    .reset_index(drop=True),
                ],
                axis="columns",
            )
        else:
            df = pd.concat(
                [
                    starts_df.drop(["selfies"], axis="columns")
                    .rename({"smiles": "start_smiles"}, axis="columns")
                    .reset_index(drop=True),
                    ends_df.drop(["selfies"], axis="columns")
                    .rename({"smiles": "end_smiles"}, axis="columns")
                    .reset_index(drop=True),
                ],
                axis="columns",
            )
        preprocess_series_fairseq("different_starts", tokenizer_suffix, df)
        (
            manhattans,
            euclideans,
            cosines,
            fixed_manhattans,
            fixed_euclideans,
            fixed_cosines,
        ) = get_distances(
            tokenizer_suffix,
            PROJECT_PATH / "different_starts" / tokenizer_suffix / "start",
            PROJECT_PATH / "different_starts" / tokenizer_suffix / "end",
            cuda,
        )
        for (similarity_name, similarity, fixed_similarity) in [
            ("dice", morgan_dice, fixed_morgan_dice),
            ("tanimoto", morgan_tanimoto, fixed_morgan_tanimoto),
            ("rdkit", rdkit_distance, fixed_rdkit),
        ]:
            for (distance_name, distance, fixed_distance) in [
                ("manhattan", manhattans, fixed_manhattans),
                ("euclid", euclideans, fixed_euclideans),
                ("cosine", cosines, fixed_cosines),
            ]:
                plotting_distances[similarity_name] = plotting_distances.get(
                    similarity_name, {}
                )
                plotting_distances[similarity_name][distance_name] = plotting_distances[
                    similarity_name
                ].get(distance_name, []) + [(tokenizer_suffix, distance)]

                plotting_distances_fixed[
                    similarity_name
                ] = plotting_distances_fixed.get(similarity_name, {})
                plotting_distances_fixed[similarity_name][
                    distance_name
                ] = plotting_distances_fixed[similarity_name].get(distance_name, []) + [
                    (tokenizer_suffix, fixed_distance)
                ]

                score = (
                    {
                        "tokenizer": tokenizer_suffix,
                        "distance": distance_name,
                        "similarity": similarity_name,
                    }
                    | get_correlation(similarity, distance)
                    | {
                        f"{key}_fixed": value
                        for (key, value) in get_correlation(
                            fixed_similarity, fixed_distance
                        ).items()
                    }
                )
                scores.append(score)
    for (similarity_name, similarity, fixed_similarity) in [
        ("dice", morgan_dice, fixed_morgan_dice),
        ("tanimoto", morgan_tanimoto, fixed_morgan_tanimoto),
        ("rdkit", rdkit_distance, fixed_rdkit),
    ]:
        for distance_name in ["manhattan", "euclid", "cosine"]:
            plot_correlation(
                similarity,
                plotting_distances[similarity_name][distance_name],
                Path(f"plots/correlation/{similarity_name}_{distance_name}.svg"),
            )
            plot_correlation(
                similarity,
                plotting_distances_fixed[similarity_name][distance_name],
                Path(f"plots/correlation/{similarity_name}_{distance_name}_fixed.svg"),
            )
    logging.info(f"Finished correlation for {tokenizer_suffix}")
    return pd.DataFrame(scores)


if __name__ == "__main__":
    cuda = parse_arguments(True, False, False)["cuda"]
    different_starts_measurements = find_correlations_different_starts(
        PROCESSED_PATH / "10m_deduplicated.csv", cuda=cuda
    )
    different_starts_measurements.to_csv("logs/different_starts.csv")
    synonym_measurements = measure_synonyms(cuda)
    synonym_measurements.to_csv("logs/synonyms.csv")
