from copy import deepcopy
from typing import Any, Iterable, Mapping, Optional

from genai_monitor.dependencies import EXTRAS_REQUIRE, require_extra
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser, Jsonable, is_jsonable
from genai_monitor.structures.conditioning_parsers.seed_types import SeedType


class TransformersTextGenerationConditioningParser(BaseConditioningParser):
    """Transformers Conditioning Parser.

    Transformers specfic conditioning parser that creates a Conditioning object
    from all parameters of inference that are convertible to json.
    """
    _tracked_seed_types = {SeedType.TORCH}  # Transformers uses PyTorch's random state

    def __init__(self, sample_fields_to_parsing_methods: Optional[Mapping[str, Any]] = None): # noqa: D107, ANN204
        require_extra("transformers", EXTRAS_REQUIRE)
        super().__init__(sample_fields_to_parsing_methods)

    def parse_func_arguments(self, *args, **kwargs) -> Jsonable:
        """Parses the function arguments and converts them into a jsonable object.

        Args:
            *args: The arguments of the function.
            **kwargs: The keyword arguments of the function.

        Returns:
            Jsonable: The parsed arguments.
        """
        from transformers import GenerationConfig

        parsed_arguments = {}

        outer_kwargs = deepcopy(kwargs)
        inner_kwargs = outer_kwargs.pop("kwargs", {})

        generation_config = outer_kwargs.pop("generation_config", None)
        if generation_config is not None:
            parsed_arguments.update(generation_config.to_dict())
        else:
            parsed_arguments.update(GenerationConfig().to_dict())

        parsed_arguments.update(outer_kwargs)

        if isinstance(inner_kwargs, dict):
            parsed_arguments.update(inner_kwargs)

        serialized = {param: self._serialize_object(val) for param, val in parsed_arguments.items()}

        return {k: v for k, v in serialized.items() if v is not None}

    @staticmethod
    def _serialize_object(obj: Any) -> Jsonable:
        """Transforms an object into a jsonable object."""
        import torch
        from transformers import LogitsProcessor, PreTrainedModel, StoppingCriteria

        if isinstance(obj, torch.Tensor):
            return obj.tolist()

        if isinstance(obj, (PreTrainedModel, LogitsProcessor, StoppingCriteria)):
            return obj.__class__.__name__

        if is_jsonable(obj):
            return obj

        if isinstance(obj, Iterable):
            return [TransformersTextGenerationConditioningParser._serialize_object(item) for item in obj]

        return None
