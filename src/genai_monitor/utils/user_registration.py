import getpass
import hashlib
import uuid

from loguru import logger

from genai_monitor.common.structures.data import User
from genai_monitor.db.manager import DBManager
from genai_monitor.db.schemas.tables import UserTable
from genai_monitor.structures.runtime_manager import RuntimeManager


def generate_user_hash():
    """Generate a unique hash using the MAC address of the machine.

    Returns:
        str: A unique hash generated from the MAC address of the machine.
    """

    hash_value = hashlib.sha256(getpass.getuser().encode()).hexdigest() + uuid.UUID(int=uuid.getnode()).hex[-12:]
    return hashlib.sha256(hash_value.encode()).hexdigest()


def register_user(db_manager: DBManager, runtime_manager: RuntimeManager):
    """Register a new user if not already registered.

    Args:
        db_manager: The database manager.
        runtime_manager: The runtime manager.
    """

    user_hash = generate_user_hash()

    existing_user = db_manager.search(
        UserTable,
        {"hash": user_hash},
    )

    if existing_user:
        runtime_manager.set_user_id(existing_user[0].id)
        logger.info(f"User {existing_user[0].name} found in the database.")
        return

    logger.info("User not found in the database. Registering new user.")
    username = getpass.getuser()
    user = User(name=username, hash=user_hash)
    user = db_manager.save(instance=user.to_orm())
    runtime_manager.set_user_id(user.id)
    logger.success(f"User {user.name} registered successfully.")
