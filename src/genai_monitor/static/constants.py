import os
from typing import Literal

from genai_monitor.utils.data import get_absolute_path

UNKNOWN_MODEL_HASH: Literal["<UNKNOWN>"] = "<UNKNOWN>"
EMPTY_MODEL_HASH: Literal["<EMPTY>"] = "<EMPTY>"

DEFAULT_GENAI_MONITOR_CONFIG_PATH = get_absolute_path(
    relative_path=".genai_eval_config.yaml", relative_to=os.path.abspath(__file__)
)
DEFAULT_PERSISTENCY_PATH: str = ".binaries/"
DEFAULT_DB_VERSION: str = "1.0.0"
