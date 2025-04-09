from genai_monitor.dependencies import EXTRAS_REQUIRE, OPENAI_AVAILABLE, require_extra

if OPENAI_AVAILABLE:
    import json
    from typing import List, Type, Union

    from openai.types import Completion
    from openai.types.chat import ChatCompletion
    from pydantic import ValidationError

    from genai_monitor.structures.output_parsers.base import BaseModelOutputParser, T
    from genai_monitor.utils.data_hashing import BaseType

    class OpenAIChatOutputParser(BaseModelOutputParser[Union[Completion, ChatCompletion]]):
        """An output parser that converts the responses of OpenAI API obtained by the client into samples."""

        _supported_output_types: List[Type] = [ChatCompletion, Completion]

        def __init__(self): # noqa: D107, ANN204
            require_extra("openai", EXTRAS_REQUIRE)
            super().__init__()

        def model_output_to_bytes(self, model_output: Union[Completion, ChatCompletion]) -> bytes:
            """Converts the model output to a byte representation.

            Args:
                model_output: The model output to convert.

            Returns:
                The byte representation of the model output.
            """
            return json.dumps(model_output.to_dict()).encode("utf-8")

        def bytes_to_model_output(self, databytes: bytes) -> Union[Completion, ChatCompletion]:
            """Converts a byte representation back into a model output.

            Args:
                databytes: The byte representation of the model output.

            Returns:
                The model output reconstructed from the byte representation.

            Raises:
                TypeError: If the schema of the output is not supported by the parser.
            """
            data = json.loads(databytes.decode("utf-8"))
            for output_type in self._supported_output_types:
                try:
                    return output_type(**data)
                except ValidationError:
                    continue
            raise TypeError(f"No supported pydantic schema for data {data} of type {type(data)}")

        def model_output_to_base_type(self, model_output: T) -> BaseType:
            """Converts a model output to base supported type.

            Args:
                model_output: The output of the model to convert.

            Returns:
                The data stored in the model output as one of the base supported types.

            Raises:
                TypeError: if model output is of type other than Completion/ChatCompletion.
            """
            if isinstance(model_output, Completion):
                return model_output.choices[0].text
            if isinstance(model_output, ChatCompletion):
                return model_output.choices[0].message.content

            raise TypeError(f"Unsupported model output type: {type(model_output).__name__}")
