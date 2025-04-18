# Artifact Tracking

## Overview

Artifact tracking is a powerful feature that allows you to associate additional information with your generative AI model inference outputs. It enables you to trace relationships between model inputs, outputs, and intermediate artifacts throughout the inference process.

## Key Concepts

The artifact tracking system builds upon the observable function registration framework and extends it with two primary capabilities:

1. **Forward Tracking**: Associate artifacts created before model inference with upcoming outputs
2. **Backward Tracking**: Associate artifacts with the most recently created inference outputs

## Registration API

### Forward Artifact Registration

To establish a forward tracking relationship, use `register_forward_artifact`:

```python
from genai_monitor.registration import register_forward_artifact

def get_timestamp():
    return {"time": sys.time}

register_forward_artifact(get_timestamp())

get_timestamp() # Artifact is created, awaits sample generation


```


### Backward Artifact Registration

To establish a backward tracking relationship, use `register_backward_artifact`:

```python
from genai_monitor.registration import register_backward_artifact

def get_timestamp():
    return {"time": sys.time}

register_backward_artifact(get_timestamp())

# Sample generation here

get_timestamp() # added relationship to above sample
```

This explicitly records which inputs, parameters, and contexts contributed to your model's outputs.
