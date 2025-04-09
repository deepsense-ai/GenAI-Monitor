# pylint: disable=ungrouped-imports
import hashlib
import json
import struct
from io import BytesIO
from typing import Any, NewType, Union

from typing_extensions import TypeAlias

from genai_monitor.dependencies import DIFFUSERS_AVAILABLE, TRANSFORMERS_AVAILABLE

BaseType: TypeAlias = Union[str, float, int]

Jsonable = NewType("Jsonable", Any)


def get_hash_from_jsonable(jsonable_value: Jsonable) -> str:
    """Get the hash of the jsonable value.

    Args:
        jsonable_value: The jsonable value to hash.

    Returns:
        The hash of the jsonable value.
    """
    serialized = json.dumps(jsonable_value, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


if DIFFUSERS_AVAILABLE:
    import numpy as np
    import torch
    from PIL import Image

    BaseType: TypeAlias = Union[torch.Tensor, np.ndarray, Image.Image, str, float, int]

elif TRANSFORMERS_AVAILABLE:
    import numpy as np
    import torch

    BaseType: TypeAlias = Union[torch.Tensor, np.ndarray, str, float, int]


def hash_base_type(data: "BaseType") -> str:
    """Get a SHA256 hash for base types: torch.Tensor, np.ndarray, PIL.Image.Image, str.

    Args:
        data: object of type torch.Tensor, np.ndarray, PIL.Image.Image, str.

    Raises:
        TypeError: if data is of unsupported type.

    Returns:
        A hash of the data.
    """
    if isinstance(data, str):
        hash_obj = hashlib.sha256(data.encode("utf-8"))
    elif isinstance(data, int):
        int_bytes = data.to_bytes((data.bit_length() + 7) // 8, byteorder="big", signed=True)
        hash_obj = hashlib.sha256(int_bytes)
    elif isinstance(data, float):
        float_bytes = struct.pack(">d", data)
        hash_obj = hashlib.sha256(float_bytes)
    elif (DIFFUSERS_AVAILABLE or TRANSFORMERS_AVAILABLE) and isinstance(data, np.ndarray):
        hash_obj = hashlib.sha256(data.tobytes())
    elif (DIFFUSERS_AVAILABLE or TRANSFORMERS_AVAILABLE) and isinstance(data, torch.Tensor):
        hash_obj = hashlib.sha256(data.numpy().tobytes())
    elif DIFFUSERS_AVAILABLE and isinstance(data, Image.Image):
        buffer = BytesIO()
        data.save(buffer, format="PNG")
        hash_obj = hashlib.sha256(buffer.getvalue())
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    return hash_obj.hexdigest()
