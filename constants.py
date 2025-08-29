""" Constants for project
SMILES or SELFIES, 2022
"""

import logging
from pathlib import Path
from typing import Dict, List, TypedDict

from deepchem.molnet import (
    load_bace_classification,
    load_bace_regression,
    load_bbbp,
    load_clearance,
    load_clintox,
    load_delaney,
    load_hiv,
    load_lipo,
    load_tox21,
)

# ----------------- Molecular Descriptors -----------------
DESCRIPTORS = [
    "NumHDonors",
    "NumAromaticRings",
    "NumAliphaticHeterocycles",
    "NumAromaticHeterocycles",
    "NumSaturatedHeterocycles",
    "NumHAcceptors",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "MolWt",
]

OTHERS = ["Chi0v", "Kappa1", "MolLogP", "MolMR", "QED"]

HEADER = DESCRIPTORS + OTHERS + ["SELFIES", "SELFIES_LEN", "SMILES"]

# ----------------- MoleculeNet Configuration -----------------
# from https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/utils/molnet_dataloader.py

class MolNetConfig(TypedDict):
    """Typed dictionary for MoleculeNet dataset configuration."""
    dataset_type: str
    load_fn: callable
    split: str
    trainingset_size: int
    tasks_wanted: List[str]  # Optional

MOLNET_DIRECTORY: Dict[str, MolNetConfig] = {
    "bace_classification": {
        "dataset_type": "classification",
        "load_fn": load_bace_classification,
        "split": "scaffold",
        "trainingset_size": 1210,
    },
    "bace_regression": {
        "dataset_type": "regression",
        "load_fn": load_bace_regression,
        "split": "scaffold",
        "trainingset_size": 1210,
    },
    "bbbp": {
        "dataset_type": "classification",
        "load_fn": load_bbbp,
        "split": "scaffold",
        "trainingset_size": 1631,
    },
    "clearance": {
        "dataset_type": "regression",
        "load_fn": load_clearance,
        "split": "scaffold",
        "trainingset_size": 669,
    },
    "clintox": {
        "dataset_type": "classification",
        "load_fn": load_clintox,
        "split": "scaffold",
        "tasks_wanted": ["CT_TOX"],
        "trainingset_size": 1181,
    },
    "delaney": {  # AKA esol
        "dataset_type": "regression",
        "load_fn": load_delaney,
        "split": "scaffold",
        "trainingset_size": 902,
    },
    "hiv": {
        "dataset_type": "classification",
        "load_fn": load_hiv,
        "split": "scaffold",
        "trainingset_size": 32874,
    },
    "lipo": {
        "dataset_type": "regression",
        "load_fn": load_lipo,
        "split": "scaffold",
        "trainingset_size": 3360,
    },
    "tox21": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "scaffold",
        "tasks_wanted": ["SR-p53"],
        "trainingset_size": 6264,
    },
}

# ----------------- Tokenizer Configuration -----------------
TOKENIZER_SUFFIXES = [
    "smiles_atom_isomers",
    "smiles_atom_standard",
    "smiles_trained_isomers",
    "smiles_trained_standard",
    "selfies_atom_isomers",
    "selfies_atom_standard",
    "selfies_trained_isomers",
    "selfies_trained_standard",
]

# ----------------- Path Configuration -----------------
PROJECT_PATH = Path(__file__).parent
PROCESSED_PATH = PROJECT_PATH / "processed"
TOKENIZER_PATH = PROJECT_PATH / "tokenizer"
MODEL_PATH = PROJECT_PATH / "fairseq_models"
ANALYSIS_PATH = PROJECT_PATH / "analysis"
TASK_PATH = PROJECT_PATH / "task"
PREDICTION_MODEL_PATH = PROJECT_PATH/ "prediction_models"
TASK_MODEL_PATH = Path("/data2/jgut/SoS_models/")
PLOT_PATH = PROJECT_PATH / "plots"
FAIRSEQ_PREPROCESS_PATH = PROJECT_PATH / "fairseq_preprocess"
ANNOT_PATH = "/scratch/ifender/SOS_tmp/embeddings_pretrainingdata/"
# ----------------- Project Constants -----------------
PARSING_REGEX = r"(<unk>|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
SEED = 6217
VAL_SIZE = 10000
NUM_SEEDS = 5

# ----------------- Logging Configuration -----------------
LOGGING_CONFIG = {
    "level": logging.DEBUG,
    "format": "%(asctime)s %(levelname)s: %(message)s [in %(funcName)s at %(pathname)s:%(lineno)d]",
    "filename": "default_log.log",
    "logger_name": "Project-SoS",
}

logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    filename=LOGGING_CONFIG["filename"],
)

# Create a main logger instance
logger = logging.getLogger(LOGGING_CONFIG["logger_name"])
logger.setLevel(LOGGING_CONFIG["level"])