# Quickstart Guide for GenAI Monitor

GenAI Monitor provides automatic observability for your GenAI applications with zero code changes. Follow this guide to get started with minimal effort.

## Installation

Install the base package with support for your preferred AI framework:

```sh
# Choose one or more providers based on your needs
pip install "genai-monitor[openai]"     # For OpenAI API
pip install "genai-monitor[transformers]"  # For Hugging Face Transformers
pip install "genai-monitor[diffusers]"  # For Hugging Face Diffusers
pip install "genai-monitor[litellm]"    # For LiteLLM

# Or install with all providers
pip install "genai-monitor[all]"
```

## Implicit Monitoring (Zero-Code Changes)

Simply import the auto module at the beginning of your application:

```python
# Import this first to enable automatic monitoring
import genai_monitor.auto

# Then use your AI libraries as usual - no changes needed!
```

### Example with OpenAI

```python
import os
import genai_monitor.auto  # This enables monitoring automatically
from openai import OpenAI

# Your regular OpenAI code
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
question = "What is the capital of France?"

# First call - sent to OpenAI and cached in database
response = client.chat.completions.create(
    messages=[{"role": "user", "content": question}],
    model="gpt-3.5-turbo",
)
print(response.choices[0].message.content)

# Second call with same parameters - retrieved from database, no API call made!
response = client.chat.completions.create(
    messages=[{"role": "user", "content": question}],
    model="gpt-3.5-turbo",
)
print(response.choices[0].message.content)
```

### Example with Transformers

```python
import genai_monitor.auto  # This enables monitoring automatically
from transformers import pipeline

# Your regular transformers code
generator = pipeline('text-generation', model='gpt2')

# First call - runs model and caches result
result = generator("Hello, I'm a language model", max_length=30)
print(result[0]['generated_text'])

# Second call with same parameters - retrieved from database, no model inference!
result = generator("Hello, I'm a language model", max_length=30)
print(result[0]['generated_text'])
```

## How It Works

GenAI Monitor automatically:

1. Intercepts calls to supported AI frameworks
2. Stores inputs and outputs in a local database
3. Returns cached results for identical calls
4. Provides this functionality with zero application code changes

This approach dramatically reduces:

- API costs by eliminating duplicate calls
- Latency by skipping remote API calls for repeated queries
- Compute resources by caching expensive model inferences

## Next Steps

 - See how to use explicit registration for unsupported frameworks
 - Add artifact tracking to your model calls
 - Add metadata to your model calls

