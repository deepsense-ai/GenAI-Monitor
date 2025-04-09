import io
import json
import pickle
from typing import Any, Dict, List, Union

from genai_monitor.dependencies import EXTRAS_REQUIRE, TRANSFORMERS_AVAILABLE, require_extra
from genai_monitor.structures.output_parsers.base import BaseModelOutputParser

if TRANSFORMERS_AVAILABLE:
    import torch

    _TextGenerationReturnType = Union[torch.Tensor, Dict[str, List[Union[str, int]]]]


class TransformersTextGenerationParser(BaseModelOutputParser[_TextGenerationReturnType]):
    """Transformers Text Generation Output Parser.

    Parser for serialization of outputs of the following transformers classes:
    - AutoModelForCausalLM
    - TextGenerationPipeline
    """

    def __init__(self): # noqa: D107, ANN204
        require_extra("transformers", EXTRAS_REQUIRE)
        super().__init__()

    def model_output_to_bytes(self, model_output: "_TextGenerationReturnType") -> bytes:
        """Converts the model output to a byte representation.

        Args:
            model_output: The model output to convert.

        Returns:
            The byte representation of the model output.

        Raises:
            ValueError: for unsupported model type
        """
        from torch import Tensor, save

        if isinstance(model_output, Tensor):
            buffer = io.BytesIO()
            save(model_output.cpu() if model_output.device.type != "cpu" else model_output, buffer)
            return buffer.getvalue()

        if isinstance(model_output, dict):
            return json.dumps(model_output).encode("utf-8")

        raise ValueError(f"Unsupported model output type: {type(model_output).__name__}.")

    def bytes_to_model_output(self, databytes: bytes) -> "_TextGenerationReturnType":
        """Converts a byte representation back into model output.

        Args:
            databytes: The byte representation of the model output.

        Returns:
            The model output reconstructed from the byte representation.
        """
        from torch import load

        try:
            buffer = io.BytesIO(databytes)
            return load(buffer, weights_only=True)

        except (RuntimeError, ValueError, pickle.UnpicklingError):
            return json.loads(databytes.decode("utf-8"))

    @staticmethod
    def _validate_single_generation( # noqa: ANN205
        model_output: Union[Dict[str, Any], torch.Tensor, List[Any]],
    ):
        """Check if a model output contains a single generation.

        Raises:
            ValueError: if model output contains more than one generation.
        """
        from torch import Tensor

        if isinstance(model_output, Tensor) and model_output.size(0) > 1:
            raise ValueError("Model output contains more than one generation.")
        if isinstance(model_output, list) and len(model_output) > 1:
            raise ValueError("Model output contains more than one generation.")
