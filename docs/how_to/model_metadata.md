

# Model Metadata

## Overview

When working with generative AI models, it's often valuable to associate additional context or metadata with your model calls. The GenAI Monitor system provides a simple way to attach arbitrary metadata to your model calls using the `genai_monitor_metadata` parameter.

## Adding Metadata to Model Calls

### Basic Usage

To add metadata to your model call, simply include the `genai_monitor_metadata` parameter as part of your keyword arguments:

```python
response = model.generate(
    prompt="Summarize the latest research on large language models.",
    max_tokens=500,
    temperature=0.7,
    genai_monitor_metadata={
        "request_id": "12345",
        "user_id": "user_789",
        "application": "research_assistant"
    }
)
```

The metadata will be automatically captured and associated with both the request and response, without affecting the model's generation behavior.

### Metadata Structure

The `genai_monitor_metadata` parameter accepts any JSON-serializable dictionary. You can include any information relevant to your application:

```python
genai_monitor_metadata={
    # Request context
    "user_id": "user_123",
    "session_id": "abc-xyz-789",
    "timestamp": "2023-08-15T14:32:17Z",

    # Business context
    "project": "customer_support",
    "priority": "high",
    "category": "technical_issue",

    # Technical context
    "client_version": "2.1.3",
    "experiment_id": "prompt-variation-b",
    "retry_count": 0
}
```
