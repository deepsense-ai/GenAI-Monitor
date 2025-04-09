from typing import Dict, List, Union

import pytest
import torch

from genai_monitor.structures.output_parsers.transformers_text_generation import TransformersTextGenerationParser


@pytest.fixture
def parser():
    return TransformersTextGenerationParser()


@pytest.mark.parametrize(
    "model_output",
    [{"generated_text": "some sample text"}, {"generated_token_ids": [101, 102, 103]}, torch.tensor([101, 102, 103])],
)
def test_bidirectional_conversion(
    parser: TransformersTextGenerationParser,
    model_output: Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[int]]]],
):
    output_bytes = parser.model_output_to_bytes(model_output)
    assert isinstance(output_bytes, bytes), "Output should be of type bytes"
    reconstructed_output = parser.bytes_to_model_output(output_bytes)
    if isinstance(model_output, torch.Tensor):
        assert torch.equal(model_output, reconstructed_output), "Tensor outputs do not match"
    else:
        assert model_output == reconstructed_output, "Dictionary outputs do not match"
