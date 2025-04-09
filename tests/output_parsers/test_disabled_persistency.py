import sys

import torch
from loguru import logger

from genai_monitor.registration.api import register_class
from genai_monitor.utils.auto_mode_configuration import load_config
from tests.conftest import DummyPytorchModel


def test_disabled_persistency(
    container, tmp_settings_with_disabled_persistency, get_pytorch_model_params_for_registration, capsys
):
    config = load_config(
        tmp_settings_with_disabled_persistency.get("db.url")["value"], tmp_settings_with_disabled_persistency
    )
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.registration.api"])
    logger.remove()
    logger.add(sys.stdout)
    get_pytorch_model_params_for_registration["cls"] = DummyPytorchModel
    register_class(**get_pytorch_model_params_for_registration)
    x = torch.randn(1, 10)
    model_instance = DummyPytorchModel()
    _ = model_instance(x)
    _ = model_instance(x)

    out, err = capsys.readouterr()
    assert "PersistencyManager is disabled. Output will be generated." in out
