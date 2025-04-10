from genai_monitor.dependencies import EXTRAS_REQUIRE, require_extra
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser, Jsonable, is_jsonable
from genai_monitor.structures.conditioning_parsers.seed_types import SeedType


class OpenAIConditioningParser(BaseConditioningParser):
    """Parser inferring conditioning values from parameters of OpenAI().chat.completions.create()."""

    _excluded = ["extra_headers", "extra_query", "extra_body"]
    _tracked_seed_types = {SeedType.OPENAI}  # OpenAI API supports seed parameter

    def __init__(self, sample_fields_to_parsing_methods=None):  # noqa: D107, ANN204
        require_extra("openai", EXTRAS_REQUIRE)
        super().__init__(sample_fields_to_parsing_methods)

    def parse_func_arguments(self, *args, **kwargs) -> Jsonable:
        """Parsing function for OpenAI models.

        Excludes certain keys, and for any values that are not JSON-serializable,
        it converts them to strings to avoid a NotJsonableError.

        Parameters:
            *args: Arguments of the method.
            **kwargs: Keyword arguments of the method.

        Returns:
            A dictionary containing the parsed arguments.
        """
        from openai import NotGiven

        parsed_arguments = {}

        for key, value in kwargs.items():
            if key in self._excluded or isinstance(value, NotGiven):
                continue

            if not is_jsonable(value):
                parsed_arguments[key] = str(value)
            else:
                parsed_arguments[key] = value

        return parsed_arguments
