import os
import sys

import torch
from loguru import logger

from genai_monitor.registration.api import register_class
from genai_monitor.utils.auto_mode_configuration import load_config
from tests.conftest import DummyPytorchModel


def test_enabled_persistency(container, tmp_settings, get_pytorch_model_params_for_registration):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.registration.api"])
    persistency_manager = container.persistency_manager()
    get_pytorch_model_params_for_registration["cls"] = DummyPytorchModel
    register_class(**get_pytorch_model_params_for_registration)
    x = torch.randn(1, 10)
    model_instance = DummyPytorchModel()
    model_instance(torch.randn(2, 10))  # first inference creates directories if they do not exist yet

    number_of_samples_start = len(os.listdir(str(persistency_manager.path / "samples")))
    _ = model_instance(x)
    number_of_samples_first_inference = len(os.listdir(str(persistency_manager.path / "samples")))
    _ = model_instance(x)
    number_of_samples_second_inference = len(os.listdir(str(persistency_manager.path / "samples")))
    assert number_of_samples_start + 1 == number_of_samples_first_inference
    assert number_of_samples_first_inference == number_of_samples_second_inference


def test_persistency_fallback(container, tmp_settings, get_pytorch_model_params_for_registration, capsys):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    persistency_manager = container.persistency_manager()
    container.wire(["genai_monitor.registration.api"])

    logger.add(sys.stdout)
    get_pytorch_model_params_for_registration["cls"] = DummyPytorchModel
    register_class(**get_pytorch_model_params_for_registration)
    x = torch.randn(1, 10)
    model_instance = DummyPytorchModel()
    _ = model_instance(x)

    path = str(persistency_manager.path / "samples")
    files = os.listdir(path)
    for file in files:
        if file.endswith(".bin"):
            file_path = os.path.join(path, file)
            os.remove(file_path)
    _ = model_instance(x=x)
    out, err = capsys.readouterr()
    assert "Failed to load data from disk" in out
