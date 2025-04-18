import random

from genai_monitor.registration.api import register_class
from genai_monitor.utils.auto_mode_configuration import load_config


class DummyClass:
    def dummy_method(self, x: float, y: float) -> float:
        return x + y + random.random()  # noqa: S311


def test_caching_instances_for_class(container, tmp_settings, get_registration_params_for_class_cached_instances):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())

    get_registration_params_for_class_cached_instances["cls"] = DummyClass
    cached_instances = get_registration_params_for_class_cached_instances["max_unique_instances"]
    container.wire(["genai_monitor.registration.api"])
    register_class(**get_registration_params_for_class_cached_instances)

    model = DummyClass()
    first_iteration = [model.dummy_method(1, 2) for _ in range(cached_instances)]
    second_iteration = [model.dummy_method(1, 2) for _ in range(cached_instances)]
    assert first_iteration == second_iteration
    assert len(set(first_iteration)) == cached_instances
