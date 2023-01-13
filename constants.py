""" Constants for project
SMILES or SELFIES, 2022
"""
import logging
from pathlib import Path

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
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

DESCRIPTORS = [name for name, _ in Descriptors.descList]
CALCULATOR = MolecularDescriptorCalculator(DESCRIPTORS)

# from https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/utils/molnet_dataloader.py
MOLNET_DIRECTORY = {
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
    #    "qm7": {
    #        "dataset_type": "regression",
    #        "load_fn": load_qm7,
    #        "split": "random",
    #        "trainingset_size": 5470,
    #    },
    #    "sider": {
    #        "dataset_type": "classification",
    #        "load_fn": load_sider,
    #        "split": "scaffold",
    #        "trainingset_size": 1141,
    #    },
    "tox21": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "scaffold",
        "tasks_wanted": ["SR-p53"],
        "trainingset_size": 6264,
    },
}

TOKENIZER_SUFFIXES = [
    "selfies_sentencepiece",
    "smiles_sentencepiece",
    "smiles_atom",
    "selfies_atom",
]

PROJECT_PATH = Path(__file__).parent
PROCESSED_PATH = PROJECT_PATH / "processed"
TOKENIZER_PATH = PROJECT_PATH / "tokenizer"
MODEL_PATH = PROJECT_PATH / "model"
ANALYSIS_PATH = PROJECT_PATH / "analysis"
TASK_PATH = PROJECT_PATH / "task"
FAIRSEQ_PREPROCESS_PATH = PROJECT_PATH / "fairseq_preprocess"
SEED = 6217
VAL_SIZE = 10000
# ---------------- LOGGING CONSTANTS ----------------
DEFAULT_FORMATTER = "%(asctime)s %(levelname)s: %(message)s [in %(funcName)s at %(pathname)s:%(lineno)d]"
DEFAULT_LOG_FILE = PROJECT_PATH / "logs" / "default_log.log"
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_LOGGER_NAME = "Project-SoS"

logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format=DEFAULT_FORMATTER,
    filename=DEFAULT_LOG_FILE,
)
