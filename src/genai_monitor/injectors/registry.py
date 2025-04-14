import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Union

from attrs import define, field
from loguru import logger

from genai_monitor.db.manager import DBManager
from genai_monitor.injectors.wrapper import (
    ArtifactFunctionWrapper,
    ArtifactMethodWrapper,
    ArtifactWrapperFactory,
    FunctionWrapper,
    MethodWrapper,
    WrapperFactory,
    is_class_func,
)
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser
from genai_monitor.structures.output_parsers.base import BaseModelOutputParser
from genai_monitor.structures.persistency_manager import PersistencyManager
from genai_monitor.structures.runtime_manager import RuntimeManager


@define
class WrapperRegistry:
    """Registry for function wrappers.

    This class is responsible for registering and unregistering wrappers.

    Attributes:
        _wrapper_factory: Factory for creating wrappers.
        _registry: Dictionary of registered wrappers.
    """

    _wrapper_factory: WrapperFactory
    _registry: Dict[str, Union[FunctionWrapper, MethodWrapper]] = field(factory=dict)

    def register(  # noqa: PLR0913
        self,
        func: Callable,
        db_manager: DBManager,
        persistency_manager: PersistencyManager,
        runtime_manager: RuntimeManager,
        output_parser: BaseModelOutputParser,
        conditioning_parser: BaseConditioningParser,
        hashing_function: Callable[[Any], str],
        max_unique_instances: int = 1,
    ):
        """Register a function with the registry.

        Args:
            func: The function or method to register.
            db_manager: The database manager.
            persistency_manager: The persistency manager.
            runtime_manager: The runtime manager.
            output_parser: The output parser.
            conditioning_parser: The conditioning parser.
            hashing_function: The hashing function.
            max_unique_instances: The maximum number of unique sample instances for each conditioning.
        """
        func_name = func.__qualname__

        if func_name in self._registry:
            logger.warning(f"Overriding function {func_name}, which is already registered.")

        wrapper = self._wrapper_factory.create(
            func=func,
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=output_parser,
            conditioning_parser=conditioning_parser,
            hashing_function=hashing_function,
            max_unique_instances=max_unique_instances,
        )
        self._registry[func_name] = wrapper
        func_wrapped = wrapper.wrap(func=func)
        override_func_in_module(func=func, func_override=func_wrapped)
        override_func_in_imported_modules(func=func, func_override=func_wrapped)

    def unregister(self, func: Callable):
        """Unregister a function from the registry.

        Args:
            func: The function to unregister.
        """
        func_name = func.__qualname__
        wrapper = self._registry.pop(func_name, None)
        if wrapper is None:
            logger.warning(f"Function {func_name} can't be unregistered as it doesn't exist in the registry.")
            return
        override_func_in_module(func=func, func_override=wrapper.func)
        override_func_in_imported_modules(func=func, func_override=wrapper.func)

    def get_registered_list(self) -> List[str]:
        """Get a list of registered functions.

        Returns:
            A list of registered functions.
        """
        return list(self._registry.keys())


@define
class ArtifactRegistry:
    """Artifact registry.

    Attributes:
        _artifact_wrapper_factory: Factory for creating artifact wrappers.
        _registry: Dictionary of registered artifacts (both forward and backward tracking).
    """

    _artifact_wrapper_factory: ArtifactWrapperFactory
    _registry: Dict[str, Dict[str, Union[ArtifactFunctionWrapper, ArtifactMethodWrapper]]] = field(
        factory=lambda: {"forward": {}, "backward": {}}
    )

    def register(
        self,
        func: Callable,
        direction: str,
        db_manager: DBManager,
        persistency_manager: PersistencyManager,
        runtime_manager: RuntimeManager,
    ):
        """Registers an artifact function or method.

        Args:
            func: The function or method to register.
            direction: The direction of the artifact (either 'forward' or 'backward').
            db_manager: The database manager.
            persistency_manager: The persistency manager.
            runtime_manager: The runtime manager.

        Raises:
            ValueError: If the direction is not 'forward' or 'backward'.
        """
        func_name = func.__qualname__

        if func_name in self._registry[direction]:
            logger.warning(f"Overriding artifact {func_name}, which is already registered.")

        wrapper = self._artifact_wrapper_factory.create(
            func=func, db_manager=db_manager, persistency_manager=persistency_manager, runtime_manager=runtime_manager
        )
        if direction == "backward":
            func_wrapped = wrapper.wrap_backward(func=func)

        elif direction == "forward":
            func_wrapped = wrapper.wrap_forward(func=func)

        else:
            raise ValueError(f"Invalid direction: {direction} (must be 'forward' or 'backward').")

        self._registry[direction][func_name] = wrapper

        override_func_in_module(func=func, func_override=func_wrapped)
        override_func_in_imported_modules(func=func, func_override=func_wrapped)

    def unregister(self, func: Callable, direction: str):
        """Removes an artifact function or method from the registry.

        Args:
            func: The function or method to unregister.
            direction: The direction of the artifact (either 'forward' or 'backward').
        """
        func_name = func.__qualname__
        wrapper = self._registry[direction].pop(func_name, None)

        if wrapper is None:
            logger.warning(f"Artifact {func_name} can't be unregistered as it doesn't exist in the registry.")
            return
        override_func_in_module(func=func, func_override=wrapper.func)
        override_func_in_imported_modules(func=func, func_override=wrapper.func)


def override_func_in_module(func: Callable, func_override: Callable):
    """Override a function in the module where it is defined.

    Args:
        func: The function to override.
        func_override: The function to override with.
    """
    func_module = inspect.getmodule(func)
    if is_class_func(func):
        cls_name = func.__qualname__.split(".")[-2]
        func_class = getattr(func_module, cls_name)
        setattr(func_class, func.__name__, func_override)
    else:
        setattr(func_module, func.__name__, func_override)


def override_func_in_imported_modules(func: Callable, func_override: Callable):
    """Override a function in all modules where it is imported.

    Args:
        func: The function to override.
        func_override: The function to override with.
    """
    func_id = id(func)
    for module in sys.modules.values():
        if not module or not hasattr(module, "__dict__"):
            continue
        for name, obj in module.__dict__.items():
            if id(obj) == func_id:
                module.__dict__[name] = func_override
