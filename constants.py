import logging
from pathlib import Path

from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

DESCRIPTORS = [name for name, _ in Chem.Descriptors.descList]
CALCULATOR = MolecularDescriptorCalculator(DESCRIPTORS)


PROJECT_PATH = Path(__file__).parent
# ---------------- LOGGING CONSTANTS ----------------
DEFAULT_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s [in %(funcName)s at %(pathname)s:%(lineno)d]"
)
DEFAULT_LOG_FILE = PROJECT_PATH / "logs" / "default_log.log"
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_LOGGER_NAME = "Project-SoS"

logger = logging.getLogger("Project-SoS")
