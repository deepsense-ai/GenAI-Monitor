import json
from typing import Any, Union

import pytest
from openai.types import Completion, CompletionChoice, CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from genai_monitor.common.structures.data import Sample
from genai_monitor.structures.output_parsers.openai import OpenAIChatOutputParser


@pytest.fixture
def openai_output_parser() -> OpenAIChatOutputParser:
    return OpenAIChatOutputParser()


@pytest.fixture
def mock_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-AYW1rRYQGJXavaFUIj7ZqbKd84S1J",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content="Hello, World!", role="assistant"),
            )
        ],
        created=1732789635,
        model="gpt-4o-mini-2024-07-18",
        object="chat.completion",
        system_fingerprint="fp_0705bf87c0",
        usage=CompletionUsage(
            completion_tokens=89,
            prompt_tokens=25,
            total_tokens=114,
        ),
    )


@pytest.fixture
def mock_completion() -> Completion:
    return Completion(
        id="chatcmpl-AYW1rRYQGJXavaFUIj7ZqbKd84S1J",
        choices=[CompletionChoice(finish_reason="stop", index=0, text="Hello, World")],
        created=1732789635,
        model="gpt-4o-mini-2024-07-18",
        object="text_completion",
        system_fingerprint="fp_0705bf87c0",
        usage=CompletionUsage(
            completion_tokens=89,
            prompt_tokens=25,
            total_tokens=114,
        ),
    )


@pytest.fixture(params=["mock_completion", "mock_chat_completion"])
def parametrized_completion(request: Any) -> Any:
    return request.getfixturevalue(request.param)


def test_get_sample_from_model_output(
    openai_output_parser: OpenAIChatOutputParser, parametrized_completion: Union[Completion, ChatCompletion]
):
    sample = openai_output_parser.get_sample_from_model_output(parametrized_completion)
    assert isinstance(sample, Sample)
    assert sample.data is not None
    assert sample.hash is not None


def test_convert_to_bytes(
    openai_output_parser: OpenAIChatOutputParser, parametrized_completion: Union[Completion, ChatCompletion]
):
    data = parametrized_completion.to_dict()
    expected_bytes = json.dumps(data).encode("utf-8")
    assert openai_output_parser.model_output_to_bytes(parametrized_completion) == expected_bytes


def test_get_model_output_from_sample(
    openai_output_parser: OpenAIChatOutputParser, parametrized_completion: Union[Completion, ChatCompletion]
):
    sample = openai_output_parser.get_sample_from_model_output(parametrized_completion)
    reconstructed_output = openai_output_parser.get_model_output_from_sample(sample)
    output_type = type(parametrized_completion)
    assert isinstance(reconstructed_output, output_type)
    assert reconstructed_output.to_dict() == parametrized_completion.to_dict()
    assert reconstructed_output == parametrized_completion


def test_unsupported_model_output_type(openai_output_parser: OpenAIChatOutputParser):
    data = json.dumps({"data": "Hello, world!"}).encode("utf-8")
    sample = Sample(data=data)
    with pytest.raises(expected_exception=TypeError):
        openai_output_parser.get_model_output_from_sample(sample=sample)
