from conftest import dummy_callable_func

from genai_monitor.common.structures.data import Sample
from genai_monitor.database_versioning import set_database_version, set_runtime_version
from genai_monitor.db.schemas.tables import SampleTable
from genai_monitor.registration.api import register_function
from genai_monitor.utils.auto_mode_configuration import load_config


def test_sample_versioning(container, tmp_settings, get_registration_params_for_callable):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.database_versioning", "genai_monitor.registration.api"])
    registration_params = get_registration_params_for_callable
    registration_params["func"] = dummy_callable_func
    register_function(**registration_params)

    db_manager = container.db_manager()

    n_samples = len(db_manager.search(SampleTable))
    set_database_version("test_sample_versioning_0")
    _ = dummy_callable_func(1, 2)
    _ = dummy_callable_func(1, 2)
    set_database_version("test_sample_versioning_1")
    _ = dummy_callable_func(1, 2)
    _ = dummy_callable_func(1, 2)
    set_runtime_version("test_sample_versioning_2")
    _ = dummy_callable_func(1, 2)
    _ = dummy_callable_func(1, 2)

    n_samples_after = len(db_manager.search(SampleTable))
    assert n_samples_after == n_samples + 3

    samples_first_version = db_manager.search(model=SampleTable, filters={"version": "test_sample_versioning_0"})
    samples_second_version = db_manager.search(model=SampleTable, filters={"version": "test_sample_versioning_1"})
    samples_third_version = db_manager.search(model=SampleTable, filters={"version": "test_sample_versioning_2"})
    assert len(samples_first_version) == 1
    assert len(samples_second_version) == 1
    assert len(samples_third_version) == 1

    first_conditioning = Sample.from_orm(samples_first_version[0]).conditioning
    second_conditioning = Sample.from_orm(samples_second_version[0]).conditioning
    third_conditioning = Sample.from_orm(samples_third_version[0]).conditioning
    assert first_conditioning == second_conditioning
    assert second_conditioning == third_conditioning
