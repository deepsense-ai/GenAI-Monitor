import os

from loguru import logger

from genai_monitor.injectors.containers import get_container
from genai_monitor.utils.auto_mode_configuration import load_config
from genai_monitor.utils.user_registration import register_user

config = load_config(os.getenv("GENAI_EVAL_DB_URL", "sqlite:///genai_eval.db"))
logger.debug(f"Loaded configuration from database: {config}")

container = get_container()
container.config.from_dict(config.model_dump())
container.wire(["genai_monitor.registration.api"])
logger.success(f"Configured Dependency Container with values: {config}")

logger.debug("User registration module imported.")
register_user(db_manager=container.db_manager(), runtime_manager=container.runtime_manager())

# pylint: disable=wrong-import-position
from genai_monitor.registration import auto  # noqa
