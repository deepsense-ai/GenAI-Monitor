import pytest
from litellm.types.utils import (
    Choices,
    CompletionTokensDetailsWrapper,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

from genai_monitor.structures.output_parsers.litellm import LiteLLMCompletionOutputParser


@pytest.fixture
def llmlite_output_parser():
    return LiteLLMCompletionOutputParser()


@pytest.fixture
def llmlite_completion_response():
    return ModelResponse(
        id="chatcmpl-AxWb1qY6ZE9BJBZaJqbE8ySUE5Sf0",
        created=1738750015,
        model="gpt-3.5-turbo-0125",
        object="chat.completion",
        system_fingerprint=None,
        choices=[
            Choices(
                finish_reason="length",
                index=0,
                message=Message(
                    content="Hello! I'm just a computer program, so",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    provider_specific_fields={"refusal": None},
                    refusal=None,
                ),
            )
        ],
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=13,
            total_tokens=23,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=0,
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
            prompt_tokens_details=PromptTokensDetailsWrapper(
                audio_tokens=0, cached_tokens=0, text_tokens=None, image_tokens=None
            ),
        ),
        service_tier="default",
    )


def test_litellm_output_parser(llmlite_output_parser, llmlite_completion_response):
    sample = llmlite_output_parser.get_sample_from_model_output(llmlite_completion_response)
    assert sample.hash is not None
    recreated_output = llmlite_output_parser.bytes_to_model_output(sample.data)
    recreated_output_dict = recreated_output.to_dict()
    llmlite_completion_response_dict = llmlite_completion_response.to_dict()
    for key in llmlite_completion_response.to_dict():
        assert recreated_output_dict[key] == llmlite_completion_response_dict[key]
