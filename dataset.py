"""Dataset preparation for SMILES or SELFIES molecular representations.

This module handles the preprocessing of MoleculeNet datasets for fine-tuning tasks,
including tokenization and fairseq-compatible dataset preparation.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from deepchem.feat import RawFeaturizer
from constants import (
    FAIRSEQ_PREPROCESS_PATH,
    MOLNET_DIRECTORY,
    TASK_PATH,
    TOKENIZER_PATH,
    TOKENIZER_SUFFIXES,
)
from tokenisation import get_tokenizer, tokenize_dataset

# Set threading layer for MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Configure logging
logger = logging.getLogger(__name__)


def prepare_molnet(
    task: str,
    tokenizer: object,
    selfies: bool,
    output_dir: Path,
    model_dict: Path,
    big_c: bool = False,
) -> None:
    """Prepare MoleculeNet tasks for fairseq fine-tuning.

    Args:
        task: Name of the MoleculeNet task to prepare
        tokenizer: Tokenizer instance to use for processing
        selfies: Whether to use SELFIES representation (must match tokenizer)
        output_dir: Directory to save preprocessed files
        model_dict: Path to vocabulary file for preprocessing
        big_c: Whether to use big-C tokenization (for standard atom tokenizers)
    """
    molnet_info = MOLNET_DIRECTORY[task]
    embedding_name = "SELFIES" if selfies else "SMILES"
    
    # Load and split the dataset
    _, splits, _ = molnet_info["load_fn"](
        featurizer=RawFeaturizer(smiles=True),
        splitter=molnet_info["split"],
        reload=False,
    )

    # Process each split (train, valid, test)
    for split_name, split_data in zip(["train", "valid", "test"], splits):
        # Tokenize molecules
        molecules = tokenize_dataset(tokenizer, split_data.X, selfies, big_c)
        
        # Handle labels
        if "tasks_wanted" in molnet_info:
            task_idx = split_data.tasks.tolist().index(molnet_info["tasks_wanted"][0])
            labels = split_data.y[:, task_idx]
        else:
            labels = split_data.y
        
        # Filter out invalid molecules
        valid_mask = ~pd.isna(molecules)
        labels = labels[valid_mask]
        
        # Log processing statistics
        num_invalid = sum(~valid_mask)
        percent_invalid = (num_invalid / len(molecules)) * 100
        logger.info(
            f"Task {task} - {split_name} - {embedding_name}: "
            f"{num_invalid} ({percent_invalid:.2f}%) invalid samples"
        )

        molecules = molecules[valid_mask]
        
        # Save processed files
        molecules.tofile(output_dir / f"{split_name}.input", sep="\n", format="%s")
        labels.tofile(output_dir / f"{split_name}.label", sep="\n", format="%s")
        
        # Additional processing for regression tasks
        if molnet_info["dataset_type"] == "regression":
            (output_dir / "label").mkdir(parents=True, exist_ok=True)
            labels.tofile(
                output_dir / "label" / f"{split_name}.label",
                sep="\n",
                format="%s",
            )

    # Run fairseq preprocessing
    _run_fairseq_preprocessing(output_dir, model_dict)


def _run_fairseq_preprocessing(output_dir: Path, model_dict: Path) -> None:
    """Execute fairseq preprocessing commands.
    
    Args:
        output_dir: Directory containing input files
        model_dict: Path to dictionary file
    """
    input_cmd = (
        f"fairseq-preprocess --only-source "
        f"--trainpref {output_dir/'train.input'} "
        f"--validpref {output_dir/'valid.input'} "
        f"--testpref {output_dir/'test.input'} "
        f"--destdir {output_dir/'input0'} "
        f"--srcdict {model_dict} --workers 60"
    )
    os.system(input_cmd)

    label_cmd = (
        f"fairseq-preprocess --only-source "
        f"--trainpref {output_dir/'train.label'} "
        f"--validpref {output_dir/'valid.label'} "
        f"--testpref {output_dir/'test.label'} "
        f"--destdir {output_dir/'label'} --workers 60"
    )
    os.system(label_cmd)


def main() -> None:
    """Main function to process all MoleculeNet tasks with all tokenizers."""
    for tokenizer_suffix in TOKENIZER_SUFFIXES:
        selfies = tokenizer_suffix.startswith("selfies")
        tokenizer = get_tokenizer(TOKENIZER_PATH / tokenizer_suffix)
        preprocess_path = FAIRSEQ_PREPROCESS_PATH / tokenizer_suffix / "dict.txt"
        
        for task_name in MOLNET_DIRECTORY:
            # Standard processing
            output_dir = TASK_PATH / task_name / tokenizer_suffix
            output_dir.mkdir(parents=True, exist_ok=True)
            prepare_molnet(task_name, tokenizer, selfies, output_dir, preprocess_path)
            logger.info(f"Created dataset: {output_dir}")
            
            # Additional processing for standard atom tokenizers
            if tokenizer_suffix in ["smiles_atom_standard", "selfies_atom_standard"]:
                big_c_dir = TASK_PATH / task_name / f"{tokenizer_suffix}_big_c"
                big_c_dir.mkdir(parents=True, exist_ok=True)
                prepare_molnet(task_name, tokenizer, selfies, big_c_dir, preprocess_path, big_c=True)
                logger.info(f"Created dataset (big-C): {big_c_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()