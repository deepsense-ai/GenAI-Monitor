from enum import Enum, auto


class SeedType(Enum):
    """Enum representing different types of seeds that can be tracked in conditioning metadata."""

    PYTHON = auto()
    NUMPY = auto()
    TORCH = auto()
    DIFFUSERS = auto()  # For Stable Diffusion and other diffusion models
    OPENAI = auto()  # For OpenAI API calls that support seed parameter
