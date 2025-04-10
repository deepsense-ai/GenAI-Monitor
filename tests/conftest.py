import io
import random
from copy import deepcopy
from typing import Any, Dict

import pytest
import torch
from torch import nn

from genai_monitor.injectors.containers import DependencyContainer
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser, Jsonable
from genai_monitor.structures.output_parsers.base import BaseModelOutputParser
from genai_monitor.utils.model_hashing import default_model_hashing_function


@pytest.fixture
def container():
    container = DependencyContainer()
    yield container

    container.wrapper_registry()._registry.clear()
    container.reset_singletons()
    del container


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("GENAI_EVAL_DB_URL", f"sqlite:///{str(tmp_path / 'test.db')}")

    return {
        "persistency.enabled": {
            "value": "true",
            "description": "Whether persistency is enabled",
        },
        "persistency.path": {
            "value": str(tmp_path),
            "description": "Path for persistent storage",
        },
        "db.url": {
            "value": f"sqlite:///{str(tmp_path / 'test.db')}",
            "description": "Database connection URL",
        },
        "version": {
            "value": "test_version",
            "description": "Database version",
        },
    }


@pytest.fixture
def tmp_settings_with_disabled_persistency(tmp_settings):
    tmp_settings["persistency.enabled"]["value"] = "false"
    return tmp_settings


class DummyPytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def conditioning_parser_for_tensors():
    class SimpleConditioningParserForTensors(BaseConditioningParser):
        @staticmethod
        def traverse_and_covert_to_jsonable(params: Dict[str, Any]): # noqa: ANN205
            for param, param_value in params.items():
                if isinstance(param_value, torch.Tensor):
                    params[param] = param_value.tolist()
                elif isinstance(param_value, Dict):
                    SimpleConditioningParserForTensors.traverse_and_covert_to_jsonable(param_value)

        def parse_func_arguments(self, *args, **kwargs) -> Jsonable:
            parsed_arguments = deepcopy(kwargs)
            self.traverse_and_covert_to_jsonable(parsed_arguments)
            return parsed_arguments

    return SimpleConditioningParserForTensors()


@pytest.fixture
def output_parser_for_tensors():
    class DummyParserForTensors(BaseModelOutputParser[torch.Tensor]):
        def model_output_to_bytes(self, model_output: torch.Tensor) -> bytes:
            buffer = io.BytesIO()
            torch.save(model_output, buffer)
            return buffer.getvalue()

        def bytes_to_model_output(self, databytes: bytes) -> torch.Tensor:
            buffer = io.BytesIO(databytes)
            return torch.load(buffer, weights_only=True)

    return DummyParserForTensors()


@pytest.fixture
def conditioning_parser_for_torch_models():
    class SimpleConditioningParserForTorchModels(BaseConditioningParser):
        @staticmethod
        def parse_func_arguments(**kwargs) -> Jsonable:
            parsed_arguments = deepcopy(kwargs)
            for key, value in parsed_arguments.items():
                if isinstance(value, torch.Tensor):
                    parsed_arguments[key] = value.tolist()
            return parsed_arguments

    return SimpleConditioningParserForTorchModels()


@pytest.fixture
def get_pytorch_model_params_for_registration():
    def model_output_to_bytes(self, model_output: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(model_output, buffer)
        return buffer.getvalue()

    def bytes_to_model_output(self, databytes: bytes) -> torch.Tensor:
        buffer = io.BytesIO(databytes)
        return torch.load(buffer, weights_only=True)

    def parse_func_arguments(**kwargs) -> Jsonable:
        parsed_arguments = deepcopy(kwargs)
        for key, value in parsed_arguments.items():
            if isinstance(value, torch.Tensor):
                parsed_arguments[key] = value.tolist()
        return parsed_arguments

    return {
        "inference_methods": ["forward", "__call__"],
        "model_output_to_bytes": model_output_to_bytes,
        "bytes_to_model_output": bytes_to_model_output,
        "parse_inference_method_arguments": parse_func_arguments,
        "model_hashing_function": default_model_hashing_function,
        "max_unique_instances": 1,
    }


def dummy_callable_func(x: float, y: float):
    return x + y + random.random() # noqa: S311


@pytest.fixture
def get_registration_params_for_callable():
    def model_output_to_bytes(x: float) -> bytes:
        return str(x).encode()

    def bytes_to_model_output(x: bytes) -> float:
        return float(x.decode())

    def parse_inference_method_arguments(**kwargs) -> Dict[str, Any]:
        return kwargs

    return {
        "model_output_to_bytes": model_output_to_bytes,
        "bytes_to_model_output": bytes_to_model_output,
        "parse_inference_method_arguments": parse_inference_method_arguments,
        "model_hashing_function": lambda x: str(x),
        "max_unique_instances": 1,
    }


@pytest.fixture
def get_registration_params_for_callable_cached_instances(get_registration_params_for_callable):
    get_registration_params_for_callable["max_unique_instances"] = 3
    return get_registration_params_for_callable


class DummyClass:
    def __init__(self):
        pass

    def dummy_method(self, x: int, y: int, z: int) -> float:
        return x + y + z + random.random() # noqa: S311


@pytest.fixture
def get_registration_params_for_class():
    def model_output_to_bytes(x: int) -> bytes:
        return str(x).encode()

    def bytes_to_model_output(x: bytes) -> float:
        return float(x.decode())

    def parse_inference_method_arguments(**kwargs) -> Dict[str, Any]:
        return kwargs

    return {
        "inference_methods": ["dummy_method"],
        "model_output_to_bytes": model_output_to_bytes,
        "bytes_to_model_output": bytes_to_model_output,
        "parse_inference_method_arguments": parse_inference_method_arguments,
        "model_hashing_function": lambda x: str(x),
        "max_unique_instances": 1,
    }


@pytest.fixture
def get_registration_params_for_class_cached_instances():
    def model_output_to_bytes(x: float) -> bytes:
        return str(x).encode()

    def bytes_to_model_output(x: bytes) -> float:
        return float(x.decode())

    def parse_inference_method_arguments(**kwargs) -> Dict[str, Any]:
        return kwargs

    return {
        "inference_methods": ["dummy_method"],
        "model_output_to_bytes": model_output_to_bytes,
        "bytes_to_model_output": bytes_to_model_output,
        "parse_inference_method_arguments": parse_inference_method_arguments,
        "model_hashing_function": lambda x: str(x),
        "max_unique_instances": 3,
    }
