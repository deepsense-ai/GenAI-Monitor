import pytest
from litellm.types.utils import Message

from genai_monitor.structures.conditioning_parsers.litellm import LiteLLMCompletionConditioningParser


@pytest.fixture
def litellm_conditioning_parser():
    return LiteLLMCompletionConditioningParser()


@pytest.fixture
def litellm_completions_kwargs():
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"content": "Hello, how are you?", "role": "user"},
            Message(content="I'm good, how are you?", role="system"),
        ],
        "timeout": 0,
        "temperature": 1.0,
        "top_p": 2.0,
        "n": 5,
        "tools": {
            "type": "function",
            "function": {
                "name": "add_cell",
                "description": "Adds code cell to Jupyter Notebook in memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "The source code to be stored in cell."}
                    },
                    "required": ["source"],
                },
            },
        },
        "kwargs": {"num_retries": 3},
    }


def test_parse_func_arguments(litellm_conditioning_parser, litellm_completions_kwargs):
    result = litellm_conditioning_parser.parse_func_arguments(**litellm_completions_kwargs)
    assert result["model"] == "gpt-3.5-turbo"
    assert result["messages"] == [
        {"content": "Hello, how are you?", "role": "user"},
        {"content": "I'm good, how are you?", "role": "system", "tool_calls": None, "function_call": None},
    ]
    assert result["timeout"] == 0
    assert result["temperature"] == 1.0
    assert result["top_p"] == 2.0
    assert result["n"] == 5
    assert result["tools"] == {
        "type": "function",
        "function": {
            "name": "add_cell",
            "description": "Adds code cell to Jupyter Notebook in memory.",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string", "description": "The source code to be stored in cell."}},
                "required": ["source"],
            },
        },
    }
    assert result["kwargs"] == {"num_retries": 3}
