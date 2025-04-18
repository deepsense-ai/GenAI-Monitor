from dependency_injector.wiring import Provide, inject

from genai_monitor.db.manager import DBManager
from genai_monitor.db.schemas.tables import ConfigurationTable
from genai_monitor.injectors.containers import DependencyContainer
from genai_monitor.structures.runtime_manager import RuntimeManager


@inject
def set_database_version(version: str, db_manager: DBManager = Provide[DependencyContainer.db_manager]):
    """Set the database version.

    Args:
        version: The version to set.
        db_manager: The database manager.
    """
    db_manager.update(model=ConfigurationTable, filters={"key": "version"}, values={"value": version})


@inject
def get_database_version(db_manager: DBManager = Provide[DependencyContainer.db_manager]) -> str:
    """Get the current database version.

    Args:
        db_manager: The database manager.

    Returns:
        str: The current database version

    Raises:
        ValueError: If the database version is not found or multiple versions are found.
    """
    version = db_manager.search(ConfigurationTable, {"key": "version"})

    if not version:
        raise ValueError("Database version not found.")

    if len(version) > 1:
        raise ValueError("Multiple database versions found.")

    return version[0].to_dict().get("value")


@inject
def set_runtime_version(version: str, runtime_manager: RuntimeManager = Provide[DependencyContainer.runtime_manager]):
    """Set the runtime version.

    Args:
        version: The version to set.
        runtime_manager: The runtime manager.
    """
    runtime_manager.set_runtime_version(version)


@inject
def get_current_version(
    runtime_manager: RuntimeManager = Provide[DependencyContainer.runtime_manager],
    db_manager: DBManager = Provide[DependencyContainer.db_manager],
) -> str:
    """Get the current version.

    Args:
        runtime_manager: The runtime manager.
        db_manager: The database manager.

    Returns:
        str: The current version.
    """
    if runtime_manager.version is not None:
        return runtime_manager.version

    return get_database_version(db_manager=db_manager)
