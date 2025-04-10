import pytest
import torch
from PIL import Image

from genai_monitor.structures.conditioning_parsers.stable_diffusion import StableDiffusionConditioningParser


@pytest.fixture
def stable_diffusion_conditioning_parser():
    return StableDiffusionConditioningParser()


@pytest.fixture
def diffusers_kwargs():
    return {
        "prompt": "example_prompt",
        "prompt2": "example_prompt_2",
        "image": Image.new("RGB", (1024, 1024)),
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "timesteps": [1, 2, 3],
        "sigmas": None,
        "guidance_scale": 5.0,
        "negative_prompt": "negative_prompt",
        "negative_prompt_2": "negative_prompt_2",
        "num_images_per_prompt": 1,
        "eta": 0.0,
        "generator": torch.Generator().manual_seed(42),
        "latents": torch.Tensor([1, 2, 3]),
        "prompt_embeds": torch.Tensor([1, 2, 3]),
        "negative_prompt_embeds": torch.Tensor([1, 2, 3]),
        "pooled_prompt_embeds": torch.Tensor([1, 2, 3]),
        "negative_pooled_prompt_embeds": torch.Tensor([1, 2, 3]),
        "ip_adapter_image": None,
        "ip_adapter_image_embeds": None,
        "output_type": "pil",
        "cross_attention_kwargs": None,
        "controlnet_conditioning_scale": 0.5,
        "guess_mode": False,
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
        "original_size": None,
        "crops_coords_top_left": (0, 0),
        "target_size": None,
        "negative_original_size": None,
        "negative_crops_coords_top_left": (0, 0),
        "negative_target_size": None,
        "clip_skip": None,
        "callback_on_step_end": None,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }


def test_parse_func_arguments_included_given_fields(stable_diffusion_conditioning_parser, diffusers_kwargs):
    result = stable_diffusion_conditioning_parser.parse_func_arguments(**diffusers_kwargs)
    assert result["prompt"] == "example_prompt"
    assert result["prompt2"] == "example_prompt_2"
    assert result["height"] == 1024
    assert result["width"] == 1024
    assert result["num_inference_steps"] == 50
    assert result["timesteps"] == [1, 2, 3]
    assert result["guidance_scale"] == 5.0
    assert result["negative_prompt"] == "negative_prompt"
    assert result["negative_prompt_2"] == "negative_prompt_2"
    assert result["num_images_per_prompt"] == 1
    assert result["eta"] == 0.0
    assert result["generator"] == 316607
    assert result["latents"] == [1.0, 2.0, 3.0]
    assert result["prompt_embeds"] == [1.0, 2.0, 3.0]
    assert result["negative_prompt_embeds"] == [1.0, 2.0, 3.0]
    assert result["pooled_prompt_embeds"] == [1.0, 2.0, 3.0]
    assert result["negative_pooled_prompt_embeds"] == [1.0, 2.0, 3.0]
    assert result["output_type"] == "pil"
    assert result["controlnet_conditioning_scale"] == 0.5
    assert result["guess_mode"] is False
    assert result["control_guidance_start"] == 0.0
    assert result["control_guidance_end"] == 1.0
    assert result["crops_coords_top_left"] == (0, 0)
    assert result["negative_crops_coords_top_left"] == (0, 0)
    assert result["callback_on_step_end_tensor_inputs"] == ["latents"]
