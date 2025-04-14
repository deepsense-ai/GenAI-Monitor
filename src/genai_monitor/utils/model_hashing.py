from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from loguru import logger

from genai_monitor.dependencies import DIFFUSERS_AVAILABLE, TRANSFORMERS_AVAILABLE
from genai_monitor.static.constants import EMPTY_MODEL_HASH, UNKNOWN_MODEL_HASH

if TRANSFORMERS_AVAILABLE:
    from torch import nn
    from transformers import PreTrainedModel

    def get_transformers_model_hash(transformers_model: PreTrainedModel, hasher: Optional[hashlib._Hash] = None) -> str:
        """Hashes the weights of a transformers model.

        Args:
            transformers_model: The transformers model to hash.
            hasher: The hasher object.

        Returns:
            str: The hash of the model.
        """
        if not hasher:
            hasher = hashlib.sha256()

        for module in transformers_model.modules():  # type: ignore
            if isinstance(module, nn.Module):
                get_pytorch_model_hash(module)
            else:
                get_transformers_model_hash(module, hasher)

        return hasher.hexdigest()

    def get_pytorch_model_hash(module: nn.Module, hasher: Optional[hashlib._Hash] = None) -> str:
        """Hashes the weights of a Pytorch nn.Module.

        Args:
            module: The module to hash.
            hasher: The hasher object.

        Returns:
            str: The hash of the model.
        """
        if not hasher:
            hasher = hashlib.sha256()
        for param_name, param in module.state_dict().items():
            if param.device == "meta":
                logger.debug(f"Skipping meta tensor {param_name}")
                continue
            hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()


if DIFFUSERS_AVAILABLE:
    from diffusers import DiffusionPipeline
    from torch import nn

    def get_diffusers_model_hash(diffusers_model: DiffusionPipeline) -> str:
        """Hashes the components of a DiffusionPipeline class object.

        Args:
            diffusers_model: The pipeline to hash.

        Returns:
            str: The hash of the model.
        """
        hasher = hashlib.sha256()
        traverse_and_hash_component(diffusers_model, hasher)
        return hasher.hexdigest()

    def hash_pipeline_components(
        pipeline: DiffusionPipeline,
        hasher: hashlib._Hash | None = None,
        verbose: bool = True,
    ):
        """Hashes the components of a DiffusionPipeline class object.

        Args:
            pipeline: The pipeline to hash.
            hasher: The hasher object.
            verbose: Whether to log the progress.
        """
        if not hasher:
            hasher = hashlib.sha256()

        if hasattr(pipeline, "components"):
            for component_name in sorted(pipeline.components.keys()):
                if verbose:
                    logger.info(f"Processing component {component_name}")
                traverse_and_hash_component(component=pipeline.components[component_name], hasher=hasher)

        else:
            for module in pipeline.modules():  # type: ignore
                traverse_and_hash_component(component=module, hasher=hasher)

    def traverse_and_hash_component(component: Any, hasher: hashlib._Hash):
        """Traverses and hashes a component.

        Args:
            component: The component to hash.
            hasher: The hasher object.
        """
        if isinstance(component, nn.Module):
            get_pytorch_model_hash(component, hasher)

        elif isinstance(component, dict):
            for subcomponent_key in sorted(component.keys()):
                traverse_and_hash_component(component=component[subcomponent_key], hasher=hasher)

        elif isinstance(component, list):
            for subcomponent in component:
                traverse_and_hash_component(component=subcomponent, hasher=hasher)


def default_model_hashing_function(model: Any) -> str | Literal[UNKNOWN_MODEL_HASH]:  # type: ignore
    """Default model hashing function.

    Args:
        model: The model to hash.

    Returns:
        str | Literal[UNKNOWN_MODEL_HASH]: The hash of the model.
    """
    try:
        return get_component_hash(model)
    except Exception as e:
        logger.error(f"Failed to hash model: {e}")
        return UNKNOWN_MODEL_HASH


def get_component_hash(component: Any) -> str:
    """Creates a hash of a component by including all instance attributes.

    Excludes private attributes (starting with '_') and callable methods.
    Normalizes paths to make the hash machine-independent.

    Args:
        component: The component to hash

    Returns:
        str: Hexadecimal hash string representing the component's configuration
    """
    hasher = hashlib.sha256()

    attributes = {
        name: value for name, value in vars(component).items() if not name.startswith("_") and not callable(value)
    }

    for attr_name, value in sorted(attributes.items()):
        hasher.update(attr_name.encode())

        if value is None:
            hasher.update(b"None")

        elif isinstance(value, (list, dict)):
            try:
                hasher.update(json.dumps(value, sort_keys=True).encode())
            except (TypeError, ValueError) as e:
                logger.warning(f"Attribute '{attr_name}' with value '{value}' is not JSON serializable: {e}")
                hasher.update(str(value).encode())

        elif isinstance(value, PathLike) or (isinstance(value, str) and ("/" in value or "\\" in value)):
            hasher.update(Path(value).name.encode())

        else:
            hasher.update(str(value).encode())

    return hasher.hexdigest()


def get_empty_model_hash(*args, **kwargs) -> Literal["<EMPTY>"]:
    """Returns an empty model hash, e.g. for API based models that should not be identified by the hash.

    Args:
        args: provided for compatibility with other hashing methods
        kwargs: provided for compatibility with other hashing methods

    Returns:
        a string denoting an empty model hash.
    """
    return EMPTY_MODEL_HASH


def get_function_full_path(func: Callable):
    """Get the full module path and function name.

    Args:
        func: The function to analyze

    Returns:
        str: Full path in format module.function_name
    """
    module_name = func.__module__
    func_name = func.__name__
    return f"{module_name}.{func_name}"
