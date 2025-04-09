# Custom Registration for GenAI Monitor

The GenAI Monitor registration API allows you to extend monitoring capabilities to any function or class of your choice, even if they're not automatically supported by the built-in integrations.

## How Custom Registration Works

When you register a function or class method:

1. GenAI Monitor intercepts calls to the registered function/method
2. Input arguments are captured and hashed for lookup
3. If identical inputs were previously processed, results are retrieved from the database
4. If this is a new input, the original function executes and results are stored
5. All of this happens transparently to your application code

## Key Registration Parameters

- `model_output_to_bytes`: Converts any model output to `bytes` type
- `bytes_to_model_output`: Converts stored `bytes` object back to the original output format
- `parse_inference_method_arguments`: Parses call arguments into a jsonable format for caching and lookup
- `sample_fields_to_parsing_methods`: Maps input fields to parsing functions (# TODO: is this still relevant?)
- `max_unique_instances`: Controls how many unique outputs to store per input (# add section on this + add link here)

This flexible registration system allows you to bring GenAI Monitor's observability to any AI component in your system, whether it's a third-party library or your own custom implementation.


## Registering Custom Classes

Use `register_class` to monitor inference methods of any Python class:


```python
from copy import deepcopy
import io
from torch import nn
import torch

from genai_monitor.registration.api import register_class
from genai_monitor.utils.data_hashing import Jsonable
from genai_monitor.utils.model_hashing import default_model_hashing_function

import genai_monitor.auto

class DummyPytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Potentially large PyTorch model...
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def model_output_to_bytes(model_output: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(model_output, buffer)
    return buffer.getvalue()

def bytes_to_model_output(databytes: bytes) -> torch.Tensor:
    buffer = io.BytesIO(databytes)
    return torch.load(buffer, weights_only=True)

def parse_func_arguments(**kwargs) -> Jsonable:
    parsed_arguments = deepcopy(kwargs)
    for key, value in parsed_arguments.items():
        if isinstance(value, torch.Tensor):
            parsed_arguments[key] = value.tolist()
    return parsed_arguments


register_class(
    cls=DummyPytorchModel,
    inference_methods=["forward"],
    model_output_to_bytes=model_output_to_bytes,
    bytes_to_model_output=bytes_to_model_output,
    parse_inference_method_arguments=parse_func_arguments,
    model_hashing_function=default_model_hashing_function,
)

model_instance = DummyPytorchModel()
x = torch.randn(1, 10)

# First model inference is generated.
first_inference = model_instance(x)

# When model is called with the same input parameters, the output is retrieved from GenAI Monitor.
second_inference = model_instance(x)
```
