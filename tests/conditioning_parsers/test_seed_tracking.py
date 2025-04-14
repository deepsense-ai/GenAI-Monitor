import torch

from genai_monitor.structures.conditioning_parsers.openai import OpenAIConditioningParser
from genai_monitor.structures.conditioning_parsers.seed_types import SeedType
from genai_monitor.structures.conditioning_parsers.stable_diffusion import StableDiffusionConditioningParser
from genai_monitor.structures.conditioning_parsers.transformers_text_generation import (
    TransformersTextGenerationConditioningParser,
)


def test_stable_diffusion_seed_tracking():
    """Test that StableDiffusionConditioningParser correctly tracks seeds."""
    parser = StableDiffusionConditioningParser()

    # Create a torch generator with a known seed
    generator = torch.Generator()
    seed = 42
    generator.manual_seed(seed)

    # Parse conditioning with the generator
    conditioning, _ = parser.parse_conditioning(
        lambda generator: None,  # dummy function
        generator=generator,
    )

    # Check that seeds are correctly stored in metadata
    assert conditioning.value_metadata is not None
    assert "seeds" in conditioning.value_metadata
    assert SeedType.TORCH.name in conditioning.value_metadata["seeds"]
    assert conditioning.value_metadata["seeds"][SeedType.TORCH.name] == sum(generator.get_state().tolist())


def test_transformers_seed_tracking():
    """Test that TransformersTextGenerationConditioningParser correctly tracks seeds."""
    parser = TransformersTextGenerationConditioningParser()

    # Create a torch generator with a known seed
    generator = torch.Generator()
    seed = 42
    generator.manual_seed(seed)

    # Parse conditioning with the generator
    conditioning = parser.parse_conditioning(
        lambda generator: None,  # dummy function
        generator=generator,
    )

    # Check that seeds are correctly stored in metadata
    assert conditioning.value_metadata is not None
    assert "seeds" in conditioning.value_metadata
    assert SeedType.TORCH.name in conditioning.value_metadata["seeds"]
    assert conditioning.value_metadata["seeds"][SeedType.TORCH.name] == sum(generator.get_state().tolist())


def test_openai_seed_tracking():
    """Test that OpenAIConditioningParser correctly tracks seeds."""
    parser = OpenAIConditioningParser()

    # Parse conditioning with a seed parameter
    seed = 12345
    conditioning = parser.parse_conditioning(
        lambda seed: None,  # dummy function
        seed=seed,
    )

    # Check that seeds are correctly stored in metadata
    assert conditioning.value_metadata is not None
    assert "seeds" in conditioning.value_metadata
    assert SeedType.OPENAI.name in conditioning.value_metadata["seeds"]
    assert conditioning.value_metadata["seeds"][SeedType.OPENAI.name] == seed


def test_no_seed_tracking():
    """Test that parsers handle cases with no seeds correctly."""
    parsers = [
        StableDiffusionConditioningParser(),
        TransformersTextGenerationConditioningParser(),
        OpenAIConditioningParser(),
    ]

    for parser in parsers:
        # Parse conditioning without any seed parameters
        conditioning = parser.parse_conditioning(
            lambda: None,  # dummy function
        )

        # Check that metadata either doesn't exist or doesn't have seeds
        if conditioning.value_metadata is not None:
            assert "seeds" not in conditioning.value_metadata or not conditioning.value_metadata["seeds"]
