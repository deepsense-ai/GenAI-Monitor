# pylint: disable=import-outside-toplevel, redefined-outer-name, ungrouped-imports

from copy import deepcopy
from typing import Any

from genai_monitor.dependencies import EXTRAS_REQUIRE, require_extra
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser, Jsonable, is_jsonable


class LiteLLMCompletionConditioningParser(BaseConditioningParser):
    """Conditioning parser for the Lite LLM completion calls."""

    def __init__(self, sample_fields_to_parsing_methods=None):
        require_extra("litellm", EXTRAS_REQUIRE)
        super().__init__(sample_fields_to_parsing_methods)

    # pylint: disable=R1710
    def traverse_and_covert_to_jsonable(self, params: Any) -> Jsonable:
        """Recursively traverse the params and convert them to jsonable objects.

        Args:
            params: The parameters to convert.

        Returns:
            Jsonable: The converted parameters.
        """
        from litellm.types.utils import Message

        if is_jsonable(params):
            return params
        if isinstance(params, Message):
            return self.traverse_and_covert_to_jsonable(params.to_dict())
        if callable(params):
            return params.__name__ + "_" + str(hash(params))
        if isinstance(params, dict):
            return {k: self.traverse_and_covert_to_jsonable(v) for k, v in params.items()}
        if isinstance(params, list):
            return [self.traverse_and_covert_to_jsonable(item) for item in params]

    def parse_func_arguments(self, *args, **kwargs) -> Jsonable:
        """Parses the function arguments and converts them into a jsonable object.

        Args:
            *args: The arguments of the function.
            **kwargs: The keyword arguments of the function.

        Returns:
            Jsonable: The parsed arguments.
        """
        parsed_arguments = deepcopy(kwargs)
        parsed_arguments = self.traverse_and_covert_to_jsonable(parsed_arguments)
        return parsed_arguments
