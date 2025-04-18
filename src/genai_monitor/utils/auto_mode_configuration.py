import os
from datetime import datetime
from typing import Dict

from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from genai_monitor.config import Config
from genai_monitor.db import init_db
from genai_monitor.db.schemas.tables import ConfigurationTable
from genai_monitor.static.constants import DEFAULT_DB_VERSION, DEFAULT_PERSISTENCY_PATH


def _get_db_url() -> str:
    """Get database URL from environment variable or use default.

    Returns:
        str: The database URL.
    """
    return os.getenv("GENAI_MONITOR_DB_URL", "sqlite:///genai_monitor.db")


def _get_persistency_path() -> str:
    """Get persistency path from environment variable or fallback to the default if not set.

    Returns:
        str: The persistency path.
    """
    return os.getenv("GENAI_MONITOR_PERSISTENCY_PATH", DEFAULT_PERSISTENCY_PATH)


def _get_database_version() -> str:
    """Get the current database version.

    Returns:
        str: The current database version.
    """
    return os.getenv("GENAI_MONITOR_DB_VERSION", DEFAULT_DB_VERSION)


# pylint: disable=W0102
DEFAULT_SETTINGS = {
    "persistency.enabled": {
        "value": "true",
        "description": "Whether persistency is enabled",
    },
    "persistency.path": {
        "value": _get_persistency_path(),
        "description": "Path for persistent storage",
    },
    "db.url": {
        "value": _get_db_url(),
        "description": "Database connection URL",
    },
    "version": {
        "value": _get_database_version(),
        "description": "Database version",
    },
}


def _init_default_config(session: Session, default_settings: dict) -> None:
    """Initialize default configuration values in the database if they don't exist.

    Args:
        session: The database session
        default_settings: The default settings to initialize session.
    """
    current_time = datetime.utcnow().isoformat()

    for key, setting in default_settings.items():
        existing = session.query(ConfigurationTable).filter_by(key=key).first()
        if not existing:
            new_entry = ConfigurationTable(
                key=key,
                value=setting["value"],
                description=setting["description"],
                updated_at=current_time,
                is_default=True,
                default_value=setting["value"],
            )
            session.add(new_entry)

    try:
        session.commit()
    except SQLAlchemyError as e:
        logger.warning(f"Failed to initialize default config: {e}")
        session.rollback()


def load_config(db_url: str, default_settings: Dict[str, dict[str, str]] = DEFAULT_SETTINGS) -> Config:
    """Load configuration from database.

    Args:
        db_url: The database URL.
        default_settings: The default settings to use if no configuration is found.

    Returns:
        The configuration as a dictionary.

    Raises:
        RuntimeError: If the configuration could not be loaded.
    """
    logger.info(f"Using database at: {db_url}")

    # Initialize database with the url
    session_manager = init_db(database_url=db_url)

    try:
        with session_manager.session_scope() as session:
            # Initialize default configuration if needed
            _init_default_config(session, default_settings)

            # Load configuration, preferring non-default values over defaults
            config_entries = (
                session.query(ConfigurationTable).order_by(ConfigurationTable.is_default).all()  # Non-defaults first
            )

            # Build config dict, letting non-default values override defaults
            settings_dict = {}
            for entry in config_entries:
                if entry.key not in settings_dict:  # Only set if not already set by a non-default
                    settings_dict[entry.key] = entry.value

            return Config(
                persistency={  # type: ignore
                    "enabled": settings_dict["persistency.enabled"].lower() == "true",
                    "path": settings_dict["persistency.path"],
                },
                db={"url": settings_dict["db.url"]},  # type: ignore
                version=settings_dict["version"],
            )

    except SQLAlchemyError as e:
        logger.error(f"Failed to load configuration from database: {e}")
        raise RuntimeError("Failed to load configuration") from e
