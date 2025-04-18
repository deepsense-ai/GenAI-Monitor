from genai_monitor.dependencies import EXTRAS_REQUIRE, LITELLM_AVAILABLE, require_extra

if LITELLM_AVAILABLE:
    import json

    from litellm.types.utils import ModelResponse

    from genai_monitor.structures.output_parsers.base import BaseModelOutputParser

    class LiteLLMCompletionOutputParser(BaseModelOutputParser):
        """Output parser for the Lite LLM completion calls."""

        def __init__(self):  # noqa: D107, ANN204
            require_extra("litellm", EXTRAS_REQUIRE)
            super().__init__()

        def bytes_to_model_output(self, databytes: bytes) -> ModelResponse:
            """Converts a byte representation back into model output.

            Args:
                databytes: The byte representation of the model output.

            Returns:
                The model output reconstructed from the byte representation.
            """
            response = json.loads(databytes)
            model_response = ModelResponse(**response["content"])
            model_response._hidden_params = response["hidden_params"]
            return model_response

        def model_output_to_bytes(self, model_output: ModelResponse) -> bytes:
            """Converts the model output to a byte representation.

            Args:
                model_output: LiteLLM Completion output

            Returns:
                The byte representation of the model output.
            """
            return json.dumps(
                {
                    "content": model_output.to_dict(),
                    "hidden_params": model_output._hidden_params,
                }
            ).encode("utf-8")
