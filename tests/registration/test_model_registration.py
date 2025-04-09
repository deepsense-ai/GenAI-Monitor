import torch

from genai_monitor.db.schemas.tables import SampleTable
from genai_monitor.registration.api import register_class
from genai_monitor.utils.auto_mode_configuration import load_config
from tests.conftest import DummyPytorchModel


def test_model_registration(container, tmp_settings, get_pytorch_model_params_for_registration):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.registration.api"])
    registration_params = get_pytorch_model_params_for_registration
    registration_params["cls"] = DummyPytorchModel
    register_class(**registration_params)

    x = torch.randn(1, 10)

    model_instance = DummyPytorchModel()

    db_manager = container.db_manager()
    samples = db_manager.search(SampleTable)
    _ = model_instance(x)
    samples_after = db_manager.search(SampleTable)

    wrapper_registry = container.wrapper_registry()
    assert len(samples_after) == len(samples) + 1
    assert len(wrapper_registry._registry.keys()) == len(registration_params["inference_methods"])
