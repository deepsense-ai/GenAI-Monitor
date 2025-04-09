import io
import sys
from copy import deepcopy

import torch
from loguru import logger

from genai_monitor.registration.api import register_function
from genai_monitor.structures.conditioning_parsers.base import Jsonable
from genai_monitor.utils.auto_mode_configuration import load_config
from genai_monitor.utils.model_hashing import default_model_hashing_function


def tensor_sum(x: torch.Tensor, y=torch.Tensor, z=torch.Tensor) -> torch.Tensor:
    return x + y + z


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


def test_function_registration(container, tmp_settings, capsys):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())

    logger.add(sys.stdout)

    container.wire(["genai_monitor.registration.api"])
    wrapper_registry = container.wrapper_registry()
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    z = torch.tensor([7, 8, 9])

    registered_before = len(wrapper_registry._registry.keys())

    register_function(
        tensor_sum,
        model_output_to_bytes=model_output_to_bytes,
        bytes_to_model_output=bytes_to_model_output,
        parse_inference_method_arguments=parse_func_arguments,
        model_hashing_function=default_model_hashing_function,
    )
    registered_after = len(wrapper_registry._registry.keys())

    assert registered_after == registered_before + 1

    first_inference = tensor_sum(x, y, z)
    second_inference = tensor_sum(x, y, z)
    out, err = capsys.readouterr()
    logger.info(out)

    assert torch.equal(first_inference, second_inference)
    assert "Loaded sample" in out
