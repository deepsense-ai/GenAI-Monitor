from genai_monitor.database_versioning import (
    get_current_version,
    get_database_version,
    set_database_version,
    set_runtime_version,
)
from genai_monitor.utils.auto_mode_configuration import load_config


def test_version_setting(container, tmp_settings):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.database_versioning"])
    db_manager, runtime_manager = container.db_manager(), container.runtime_manager()
    current_version = get_current_version(db_manager=db_manager, runtime_manager=runtime_manager)
    db_version = get_database_version(db_manager=db_manager)
    assert current_version == db_version

    new_version = "new_version"
    set_database_version(new_version, db_manager=db_manager)
    current_version = get_current_version(db_manager=db_manager, runtime_manager=runtime_manager)
    db_version = get_database_version(db_manager=db_manager)
    assert current_version == db_version

    new_runtime_version = "new_runtime_version"
    set_runtime_version(new_runtime_version, runtime_manager=runtime_manager)
    db_version = get_database_version(db_manager=db_manager)
    assert get_current_version(db_manager=db_manager, runtime_manager=runtime_manager) == new_runtime_version
    assert db_version != new_runtime_version

    new_db_version = "new_db_version"
    set_database_version(new_db_version, db_manager=db_manager)
    assert get_database_version(db_manager=db_manager) == new_db_version
    assert get_current_version(db_manager=db_manager, runtime_manager=runtime_manager) == new_runtime_version
