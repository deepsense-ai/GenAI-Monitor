# Cached Instances

## Overview

GenAI Monitor by default stores only one copy of the same inference (identical input arguments) to optimize storage and prevent redundant API calls. However, there are many scenarios where you might want to preserve multiple unique outputs for the same input, such as:

- Exploring the variety of possible responses with non-deterministic generation
- A/B testing different outputs for the same prompt
- Building datasets with diverse completions
- Testing model consistency across multiple runs

## How It Works

### Storage Phase

When `max_unique_instances` is set to a value greater than 1:

1. For each new invocation with identical inputs, GenAI Monitor checks if the maximum number of unique instances has been reached
2. If the limit hasn't been reached, the function executes normally and the new output is stored
3. Once the limit is reached, GenAI Monitor switches to retrieval mode

### Retrieval Phase

When the maximum number of unique instances has been stored:
1. GenAI Monitor selects one of the previously stored outputs using round-robin selection
2. The selected output is returned without re-invoking the function
3. Each subsequent call with identical inputs receives the next stored output in sequence



## Configuring Multiple Unique Instances

GenAI Monitor provides registration API that supports the `max_unique_instances` parameter:

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
    max_unique_instances=3
)

model_instance = DummyPytorchModel()
x = torch.randn(1, 10)

# Below model outputs will be generated.
y1 = model_instance(x)
y2 = model_instance(x)
y3 = model_instance(x)

# Since `max_unique_instaces`=3, next generations will be retrieved from GenAI Monitor.
y4 = model_instance(x) # same as y1
y5 = model_instance(x) # same as y2
y6 = model_instance(x) # same as y3


```