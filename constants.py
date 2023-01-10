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
    load_pcba,
    load_qm7,
    load_qm8,
    load_qm9,
    load_sider,
    load_tox21,
)
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

DESCRIPTORS = [name for name, _ in Descriptors.descList]
CALCULATOR = MolecularDescriptorCalculator(DESCRIPTORS)

# from https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/utils/molnet_dataloader.py
MOLNET_DIRECTORY = {
    "hiv": {
        "dataset_type": "classification",
        "load_fn": load_hiv,
        "split": "scaffold",
    },
    "bace_classification": {
        "dataset_type": "classification",
        "load_fn": load_bace_classification,
        "split": "scaffold",
    },
    "bace_regression": {
        "dataset_type": "regression",
        "load_fn": load_bace_regression,
        "split": "scaffold",
    },
    "bbbp": {
        "dataset_type": "classification",
        "load_fn": load_bbbp,
        "split": "scaffold",
    },
    "clearance": {
        "dataset_type": "regression",
        "load_fn": load_clearance,
        "split": "scaffold",
    },
    "clintox": {
        "dataset_type": "classification",
        "load_fn": load_clintox,
        "split": "scaffold",
        "tasks_wanted": ["CT_TOX"],
    },
    "delaney": {
        "dataset_type": "regression",
        "load_fn": load_delaney,
        "split": "scaffold",
    },
    # pcba is very large and breaks the dataloader
    "pcba": {
        "dataset_type": "classification",
        "load_fn": load_pcba,
        "split": "scaffold",
    },
    "lipo": {
        "dataset_type": "regression",
        "load_fn": load_lipo,
        "split": "scaffold",
    },
    "qm7": {
        "dataset_type": "regression",
        "load_fn": load_qm7,
        "split": "random",
    },
    "qm8": {
        "dataset_type": "regression",
        "load_fn": load_qm8,
        "split": "random",
    },
    "qm9": {
        "dataset_type": "regression",
        "load_fn": load_qm9,
        "split": "random",
    },
    "sider": {
        "dataset_type": "classification",
        "load_fn": load_sider,
        "split": "scaffold",
    },
    "tox21": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "scaffold",
        "tasks_wanted": ["SR-p53"],
    },
}

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
