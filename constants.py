""" Constants for project
SMILES or SELFIES, 2022
"""
import logging
from pathlib import Path

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

DESCRIPTORS = [name for name, _ in Descriptors.descList]
CALCULATOR = MolecularDescriptorCalculator(DESCRIPTORS)

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
