import logging
from pathlib import Path

from rdkit import Chem

DESCRIPTORS = [name for name, _ in Chem.Descriptors.descList]

PROJECT_PATH = Path(__file__).parent
PROCESSED_PATH = PROJECT_PATH / "processed"
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
