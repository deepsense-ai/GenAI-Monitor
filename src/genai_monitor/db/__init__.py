import sys

from loguru import logger

from .config import SessionManager


def init_db(database_url: str = "sqlite:///genai_monitor.db") -> SessionManager:
    """Initializes the database connection.

    Args:
        database_url: The URL of the database to connect to.

    Returns:
        A `SessionManager` object.
    """
    return SessionManager(database_url=database_url)


logger.remove(0)
logger.add(sys.stderr, format="{time} | {level} | {message}")

__all__ = ["init_db"]
