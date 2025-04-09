from copy import deepcopy

from conftest import dummy_callable_func

from genai_monitor.registration.api import register_function
from genai_monitor.utils.auto_mode_configuration import load_config


def test_caching_instances_for_callable(container, tmp_settings, get_registration_params_for_callable_cached_instances):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.registration.api"])

    registration_params = deepcopy(get_registration_params_for_callable_cached_instances)
    cached_instances = registration_params["max_unique_instances"]
    registration_params["func"] = dummy_callable_func
    register_function(**registration_params)

    first_iteration = [dummy_callable_func(1, 2) for _ in range(cached_instances)]
    second_iteration = [dummy_callable_func(1, 2) for _ in range(cached_instances)]
    assert first_iteration == second_iteration
    assert len(set(first_iteration)) == cached_instances
