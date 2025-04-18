import hashlib
import io
from typing import Any, Iterable

from genai_monitor.dependencies import EXTRAS_REQUIRE, require_extra
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser, Jsonable, is_jsonable
from genai_monitor.structures.conditioning_parsers.seed_types import SeedType


class StableDiffusionConditioningParser(BaseConditioningParser):
    """Diffusers Conditioning Parser.

    Diffusers specfic conditioning parser that creates a Conditioning object
    from all parameters of inference that are convertible to json.
    """

    _tracked_seed_types = {SeedType.TORCH, SeedType.DIFFUSERS}

    def __init__(self):  # noqa: D107, ANN204
        require_extra("diffusers", EXTRAS_REQUIRE)
        super().__init__()

    def parse_func_arguments(self, *args, **kwargs) -> Jsonable:
        """Parses the function arguments and converts them into a jsonable object.

        Args:
            *args: The arguments of the function.
            **kwargs: The keyword arguments of the function.

        Returns:
            Jsonable: The parsed arguments.
        """
        parsed_arguments = dict(kwargs)
        serialized = {param: self._serialize_object(val) for param, val in parsed_arguments.items()}
        return {k: v for k, v in serialized.items() if v is not None}  # type: ignore

    @staticmethod
    def _serialize_object(obj: Any) -> Jsonable:
        """Transforms an object into a jsonable object."""
        import numpy as np
        import torch
        from PIL import Image

        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()  # type: ignore

        if isinstance(obj, torch.Generator):
            # sum(generator_state_tensor) is how we track seeds
            return sum(obj.get_state().tolist())  # type: ignore

        if isinstance(obj, Image.Image):
            # Remove any leftover metadata
            obj.info = {}
            buffer = io.BytesIO()
            # Save with reproducible settings
            obj.save(buffer, format="PNG", optimize=False, compress_level=0)
            return hashlib.sha256(buffer.getvalue()).hexdigest()  # type: ignore
        if is_jsonable(obj):
            return obj

        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return [StableDiffusionConditioningParser._serialize_object(item) for item in obj]  # type: ignore

        return None  # type: ignore
