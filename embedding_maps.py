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
from fairseq_utils import (
    get_embeddings,
    load_model,
    preprocess_series_old,
    create_untrained_prediction_model,
    transform_to_prediction_model
)
from plotting import plot_representations
from preprocessing import canonize_smile, check_valid, translate_selfie
from sample_molecules import BETA_LACTAMS, STEROIDS, TROPANES, SULFONAMIDES
from tokenisation import get_tokenizer, tokenize_to_ids
from utils import parse_arguments
from fairseq.data import Dictionary

# Visualization parameters
DEFAULT_MIN_DIST = 0.5
DEFAULT_N_NEIGHBORS = 15
DEFAULT_ALPHA = 0.6
DEFAULT_OFFSET = 0


def prepare_selected_molecules(sample_amount: int = 64) -> pd.DataFrame:
    """Prepare a balanced dataset of selected molecule classes.
    
    Args:
        sample_amount: Number of molecules to sample from each class
        
    Returns:
        DataFrame containing molecules with their SMILES, SELFIES, and labels
    """
    random.seed(4)  # Fixed seed for reproducibility
    
    molecule_classes = {
        "Steroids": list(set(STEROIDS)),
        "Beta lactams": list(set(BETA_LACTAMS)),
        "Tropanes": list(set(TROPANES)),
        "Sulfonamides": list(set(SULFONAMIDES)),
    }
    molecules = []
    labels = []
    for class_name, class_molecules in molecule_classes.items():
        sampled = random.sample(class_molecules, min(sample_amount, len(class_molecules)))
        for smile in sampled:
            if not check_valid(smile):
                continue
            selfie, len_selfie = translate_selfie(smile)
            canon_smile = canonize_smile(smile)
            if len_selfie == 0:
                continue
            molecules.append((canon_smile, selfie))
            labels.append(class_name)
    dataframe = pd.DataFrame({
        "SELFIES": [m[1] for m in molecules],
        "SMILES": [m[0] for m in molecules],
        "label": labels
    })
    return dataframe


def sample_other_molecules(data_path: Path, amount: int) -> Optional[pd.DataFrame]:
    """Sample additional molecules from a reference dataset.
    
    Args:
        data_path: Path to CSV file containing molecules
        amount: Number of molecules to sample
        
    Returns:
        DataFrame with sampled molecules or None if amount <= 0
    """
    if amount <= 0:
        return None
        
    df = pd.read_csv(data_path, usecols=["SMILES", "SELFIES"])
    return (df.dropna()
             .sample(n=amount, random_state=SEED + 39775)
             .assign(label="Other molecule"))


def get_molecule_dataframe(data_path: Path) -> pd.DataFrame:
    """Create combined dataframe of selected and background molecules.
    
    Args:
        data_path: Path to reference dataset for background molecules
        
    Returns:
        Combined DataFrame with all molecules
    """
    selected_df = prepare_selected_molecules(64)
    return selected_df


def process_molecules_with_model(
    molecule_df: pd.DataFrame,
    tokenizer_suffix: str,
    model_type: str,
    cuda: bool,
    output_dir: Path,
    min_dist: float = DEFAULT_MIN_DIST,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    alpha: float = DEFAULT_ALPHA,
    offset: int = DEFAULT_OFFSET,
) -> None:
    """Process molecules and generate embeddings visualization.
    
    Args:
        molecule_df: DataFrame containing molecules to process
        tokenizer_suffix: Identifier for tokenizer configuration
        model_type: Type of model to use ('random' or other)
        cuda: Whether to use GPU acceleration
        output_dir: Directory to save visualization
        min_dist: UMAP min_dist parameter
        n_neighbors: UMAP n_neighbors parameter
        alpha: Plot point transparency
        offset: Plot offset parameter
    """
    # Select representation based on tokenizer type
    use_selfies = tokenizer_suffix.startswith("selfies")
    molecule_series = molecule_df["SELFIES" if use_selfies else "SMILES"]
    
    # Initialize tokenizer and process molecules
    tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
    tokenized_molecules = tokenize_to_ids(tokenizer, molecule_series)
    
    # Set up paths and dictionaries
    fairseq_dict_path = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
    data_path = TASK_PATH / "bbbp" / tokenizer_suffix
    working_dir = PROJECT_PATH / f"embedding_mapping_{model_type}"
    
    # Preprocess and get embeddings
    preprocess_series_old(tokenized_molecules, working_dir, fairseq_dict_path)
    
    # Load or create model
    if model_type == "untrained":
        model_path = PREDICTION_MODEL_PATH / f"untrained_{tokenizer_suffix}" / "checkpoint_last.pt"
        if not model_path.exists():
            smiles = tokenizer_suffix.lower().startswith("smiles")
            create_untrained_prediction_model(model_path.parent, smiles)
    else:
        tokenizer_model_suffix = f"{tokenizer_suffix}_{model_type}"
        model_path = PREDICTION_MODEL_PATH / tokenizer_model_suffix / "checkpoint_last.pt"
        if not model_path.exists():
            transform_to_prediction_model(tokenizer_model_suffix)
    
    model = load_model(model_path, data_path, str(cuda))
    fairseq_dict = Dictionary.load(str(fairseq_dict_path))
    
    # Get embeddings and plot
    mol_dataset_path = working_dir / "train"
    embeddings = get_embeddings(model, mol_dataset_path, fairseq_dict, cuda)
    
    plot_representations(
        embeddings,
        molecule_df["label"],
        output_dir,
        min_dist,
        n_neighbors,
        alpha=alpha,
        offset=offset
    )


def main():
    """Main execution function for the visualization script."""
    args = parse_arguments(cuda=True, tokenizer=True, model_type=True)
    
    # Configuration
    cuda = args["cuda"]
    model_type = args["modeltype"]
    tokenizer_suffixes = (
        [args["tokenizer"]] 
        if args.get("tokenizer") 
        else TOKENIZER_SUFFIXES
    )
    if model_type == "untrained":
        tokenizer_suffixes = ["selfies_atom_isomers", "smiles_atom_isomers", ]

    # Prepare molecule data
    data_path = PROCESSED_PATH / "standard" / "full_deduplicated_standard.csv"
    molecule_df = get_molecule_dataframe(data_path)
    
    # Process with each tokenizer
    for tokenizer_suffix in tokenizer_suffixes:
        output_dir = PLOT_PATH / "selected_molecules_scaled"
        if model_type == "untrained":
            output_dir = output_dir / f"untrained_{tokenizer_suffix[:6]}"
        else:
            output_dir = output_dir / f"{tokenizer_suffix}_{model_type}"
        process_molecules_with_model(
            molecule_df,
            tokenizer_suffix,
            model_type,
            cuda,
            output_dir
        )


if __name__ == "__main__":
    main()