from typing import Any, Callable, Dict, List, Optional, Type

from dependency_injector.wiring import Provide, inject

from genai_monitor.db.manager import DBManager
from genai_monitor.injectors.containers import DependencyContainer
from genai_monitor.injectors.registry import ArtifactRegistry, WrapperRegistry
from genai_monitor.registration.utils import _make_cls
from genai_monitor.structures.conditioning_parsers.base import (
    BaseConditioningParser,
    DefaultConditioningParser,
    Jsonable,
)
from genai_monitor.structures.output_parsers.base import BaseModelOutputParser
from genai_monitor.structures.persistency_manager import PersistencyManager
from genai_monitor.structures.runtime_manager import RuntimeManager
from genai_monitor.utils.data_hashing import BaseType
from genai_monitor.utils.model_hashing import default_model_hashing_function


def _make_conditioning_parser_name(cls_name: str) -> str:
    return f"{cls_name}ConditioningParser"


def _make_output_parser_name(cls_name: str) -> str:
    return f"{cls_name}OutputParser"


def register_class(
    cls: Type,
    inference_methods: List[str],
    model_output_to_bytes: Callable[[Any], bytes],
    bytes_to_model_output: Callable[[bytes], Any],
    model_output_to_base_type: Optional[Callable[[Any], BaseType]] = None,
    parse_inference_method_arguments: Optional[Callable[[Dict[str, Any]], Jsonable]] = None,
    model_hashing_function: Optional[Callable[[object], str]] = None,
    sample_fields_to_parsing_methods: Optional[Dict[str, Callable]] = None,
    max_unique_instances: int = 1,
):
    """Registers a class with inference methods.

    Args:
        cls: The class to register.
        inference_methods: The names of the inference methods.
        model_output_to_bytes: The function to convert the model output to bytes.
        bytes_to_model_output: The function to convert bytes to the model output.
        model_output_to_base_type: The function to convert the model output to a base type.
        parse_inference_method_arguments: The function to parse the inference method arguments.
        model_hashing_function: The function to hash the model.
        sample_fields_to_parsing_methods: Optional mapping between the names of parameters in the inference method
        and functions that allow to extract the base type representation that can be looked up in the database using
        a hash.
        max_unique_instances: The maximum number of unique sample instances for each conditioning.
    """
    cls_name = cls.__name__
    methods_to_wrap = [getattr(cls, method_name) for method_name in inference_methods]
    if parse_inference_method_arguments is None:
        conditioning_parser = DefaultConditioningParser()
    else:
        conditioning_parser = _make_cls(
            cls_name=_make_conditioning_parser_name(cls_name),
            base=BaseConditioningParser,
            method_mapper={"parse_func_arguments": parse_inference_method_arguments},
        )()

    if sample_fields_to_parsing_methods:
        conditioning_parser.sample_fields_to_parsing_methods = sample_fields_to_parsing_methods

    output_parser_method_mapper = {
        "model_output_to_bytes": model_output_to_bytes,
        "bytes_to_model_output": bytes_to_model_output,
    }

    if model_output_to_base_type is not None:
        output_parser_method_mapper["model_output_to_base_type"] = model_output_to_base_type

    output_parser = _make_cls(
        cls_name=_make_output_parser_name(cls_name),
        base=BaseModelOutputParser,
        method_mapper=output_parser_method_mapper,
    )()

    if not model_hashing_function:
        model_hashing_function = default_model_hashing_function

    for inference_method in methods_to_wrap:
        register_inference_method(
            inference_method=inference_method,
            output_parser=output_parser,
            conditioning_parser=conditioning_parser,
            hashing_function=model_hashing_function,
            max_unique_instances=max_unique_instances,
        )


def register_function(
    func: Callable,
    model_output_to_bytes: Callable[[Any], bytes],
    bytes_to_model_output: Callable[[bytes], Any],
    model_output_to_base_type: Optional[Callable[[Any], BaseType]] = None,
    parse_inference_method_arguments: Optional[Callable[[Dict[str, Any]], Jsonable]] = None,
    model_hashing_function: Optional[Callable[[object], str]] = None,
    sample_fields_to_parsing_methods: Optional[Dict[str, Callable]] = None,
    max_unique_instances: int = 1,
):
    """Registers a function.

    Args:
        func: The function to register.
        model_output_to_bytes: The function to convert the model output to bytes.
        bytes_to_model_output: The function to convert bytes to the model output.
        model_output_to_base_type: The function to convert the model output to a base type.
        parse_inference_method_arguments: The function to parse the inference method arguments.
        model_hashing_function: The function to hash the model.
        sample_fields_to_parsing_methods: Optional mapping between the names of parameters in the inference method
        and functions that allow to extract the base type representation that can be looked up in the database using
        a hash.
        max_unique_instances: The maximum number of unique sample instances for each conditioning.
    """
    func_name = f"{func.__module__}.{func.__name__}"

    if parse_inference_method_arguments is None:
        conditioning_parser = DefaultConditioningParser()
    else:
        conditioning_parser = _make_cls(
            cls_name=_make_conditioning_parser_name(func_name),
            base=BaseConditioningParser,
            method_mapper={"parse_func_arguments": parse_inference_method_arguments},
        )()

    conditioning_parser.max_unique_instances = max_unique_instances  # type: ignore

    if sample_fields_to_parsing_methods:
        conditioning_parser.sample_fields_to_parsing_methods = sample_fields_to_parsing_methods

    output_parser_method_mapper = {
        "model_output_to_bytes": model_output_to_bytes,
        "bytes_to_model_output": bytes_to_model_output,
    }

    if model_output_to_base_type is not None:
        output_parser_method_mapper["model_output_to_base_type"] = model_output_to_base_type

    output_parser = _make_cls(
        cls_name=_make_output_parser_name(func.__name__),
        base=BaseModelOutputParser,
        method_mapper=output_parser_method_mapper,
    )()

    if not model_hashing_function:
        model_hashing_function = default_model_hashing_function

    register_inference_method(
        inference_method=func,
        output_parser=output_parser,
        conditioning_parser=conditioning_parser,
        hashing_function=model_hashing_function,
        max_unique_instances=max_unique_instances,
    )


@inject
def register_inference_method(
    inference_method: Callable,
    conditioning_parser: BaseConditioningParser,
    output_parser: BaseModelOutputParser,
    hashing_function: Callable,
    max_unique_instances: int = 1,
    wrapper_registry: WrapperRegistry = Provide[DependencyContainer.wrapper_registry],
    db_manager: DBManager = Provide[DependencyContainer.db_manager],
    persistency_manager: PersistencyManager = Provide[DependencyContainer.persistency_manager],
    runtime_manager: RuntimeManager = Provide[DependencyContainer.runtime_manager],
):
    """Registers an inference method.

    Args:
        inference_method: The inference method to register.
        conditioning_parser: The conditioning parser.
        output_parser: The output parser.
        hashing_function: The hashing function.
        max_unique_instances: The maximum number of unique sample instances for each conditioning.
        wrapper_registry: The wrapper registry.
        db_manager: The DB manager.
        persistency_manager: The persistency manager.
        runtime_manager: The runtime manager.
    """
    # Note: Save objects are saved to disk, thus conditioning parser needs a reference to the persistency manager
    conditioning_parser.persistency_manager = persistency_manager
    conditioning_parser.db_manager = db_manager

    wrapper_registry.register(
        func=inference_method,
        db_manager=db_manager,
        persistency_manager=persistency_manager,
        runtime_manager=runtime_manager,
        output_parser=output_parser,
        conditioning_parser=conditioning_parser,
        hashing_function=hashing_function,
        max_unique_instances=max_unique_instances,
    )


@inject
def unregister(func: Callable, wrapper_registry: WrapperRegistry = Provide[DependencyContainer.wrapper_registry]):
    """Unregisters a function.

    Args:
        func: The function to unregister.
        wrapper_registry: The wrapper registry.
    """
    wrapper_registry.unregister(func=func)


def register_forward_artifact(func: Callable):
    """Registers a function that creates artifacts for the incoming sample.

    Args:
        func: The function to register.
    """
    register_artifact(func=func, direction="forward")


def register_backward_artifact(func: Callable):
    """Registers a function that creates artifacts for the latest sample.

    Args:
        func: The function to register.
    """
    register_artifact(func=func, direction="backward")


@inject
def register_artifact(
    func: Callable,
    direction: str,
    artifact_registry: ArtifactRegistry = Provide[DependencyContainer.artifact_registry],
    db_manager: DBManager = Provide[DependencyContainer.db_manager],
    persistency_manager: PersistencyManager = Provide[DependencyContainer.persistency_manager],
    runtime_manager: RuntimeManager = Provide[DependencyContainer.runtime_manager],
):
    """Registers a function that creates artifacts.

    Args:
        func: The function to register.
        direction: The direction of the artifact.
        artifact_registry: The artifact registry.
        db_manager: The DB manager.
        persistency_manager: The persistency manager.
        runtime_manager: The runtime manager.
    """
    artifact_registry.register(
        func=func,
        direction=direction,
        db_manager=db_manager,
        persistency_manager=persistency_manager,
        runtime_manager=runtime_manager,
    )
