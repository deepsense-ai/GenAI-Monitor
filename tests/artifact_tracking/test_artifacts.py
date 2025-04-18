import torch
from loguru import logger

from genai_monitor.db.schemas.tables import ArtifactTable
from genai_monitor.registration.api import register_backward_artifact, register_class, register_forward_artifact
from genai_monitor.utils.auto_mode_configuration import load_config
from tests.conftest import DummyPytorchModel


class DummyForward:
    def dummy_method(self, x: int, y: int, z: int) -> int:
        return x * y * z


class DummyBackward:
    def dummy_method(self, x: int, y: int, z: int) -> int:
        return x * y * z


def dummy_func_forward(x: int, y: int) -> int:
    return x + y


def dummy_func_backward(x: int, y: int) -> int:
    return x - y


def test_artifact_registration(container, tmp_settings, get_pytorch_model_params_for_registration):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.registration.api"])

    runtime_manager = container.runtime_manager()

    register_backward_artifact(dummy_func_backward)
    register_backward_artifact(DummyBackward().dummy_method)

    register_forward_artifact(dummy_func_forward)
    register_forward_artifact(DummyForward().dummy_method)
    get_pytorch_model_params_for_registration["cls"] = DummyPytorchModel
    register_class(**get_pytorch_model_params_for_registration)

    model_instance = DummyPytorchModel()
    logger.info(f"Model instance: {model_instance}")

    dummy_func_forward(1, 2)
    DummyForward().dummy_method(1, 2, 3)

    assert len(runtime_manager.artifacts_for_next_sample) == 2

    x1 = torch.randn(1, 10)
    _ = model_instance(x1)

    assert len(runtime_manager.artifacts_for_next_sample) == 0
    latest_sample = runtime_manager.latest_sample
    assert latest_sample is not None

    dummy_func_backward(1, 2)
    DummyBackward().dummy_method(1, 2, 3)

    dummy_func_forward(10, 20)
    DummyForward().dummy_method(2, 2, 2)

    assert len(runtime_manager.artifacts_for_next_sample) == 2

    x2 = torch.randn(1, 10)
    _ = model_instance(x2)

    assert len(runtime_manager.artifacts_for_next_sample) == 0
    assert latest_sample.id != runtime_manager.latest_sample.id

    dummy_func_backward(10, 20)
    DummyBackward().dummy_method(10, 20, 30)

    db_manager = container.db_manager()

    artifacts = db_manager.search(ArtifactTable)
    assert len(artifacts) == 8
    assert len({artifact.sample_id for artifact in artifacts}) == 2
