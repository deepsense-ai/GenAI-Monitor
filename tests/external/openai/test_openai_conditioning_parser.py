import pytest
from openai import NotGiven

from genai_monitor.structures.conditioning_parsers.openai import OpenAIConditioningParser


@pytest.fixture
def openai_conditioning_parser():
    return OpenAIConditioningParser()


@pytest.fixture
def messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming.",
        },
    ]


@pytest.fixture
def openai_completions_kwargs(messages):
    return {
        "messages": messages,
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.7,
        "logit_bias": {"50256": -100},
        "frequency_penalty": NotGiven(),
        "extra_headers": {"Authorization": "Bearer token"},
    }


def test_parse_func_arguments_included_given_fields(messages, openai_conditioning_parser, openai_completions_kwargs):
    result = openai_conditioning_parser.parse_func_arguments(**openai_completions_kwargs)
    assert result["messages"] == messages
    assert result["model"] == "gpt-4o-mini-2024-07-18"
    assert result["temperature"] == 0.7
    assert result["logit_bias"] == {"50256": -100}
    assert "frequency_penalty" not in result
    assert "extra_headers" not in result
