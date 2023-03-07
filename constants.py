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
    "tox21": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "scaffold",
        "tasks_wanted": ["SR-p53"],
        "trainingset_size": 6264,
    },
}

REACTION_PREDICTION_DIRECTORY = {
    "lef": {
        "dataset_type": "generation",
        "trainingset_size": 296578,
    },
    "jin": {
        "dataset_type": "generation",
        "trainingset_size": 391412,
    },
    "schwaller": {
        "dataset_type": "generation",
        "trainingset_size": 902271,
    },
}
TOKENIZER_SUFFIXES = [
    "smiles_atom",
    "selfies_atom",
    "selfies_sentencepiece",
    "smiles_sentencepiece",
]

PROJECT_PATH = Path(__file__).parent
PROCESSED_PATH = PROJECT_PATH / "processed"
TOKENIZER_PATH = PROJECT_PATH / "tokenizer"
MODEL_PATH = PROJECT_PATH / "model"
ANALYSIS_PATH = PROJECT_PATH / "analysis"
TASK_PATH = PROJECT_PATH / "task"
USPTO_PATH = PROJECT_PATH / "download_uspto"
TASK_MODEL_PATH = Path("/data/jgut/SoS_models/")

PARSING_REGEX = r"(<unk>|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


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
