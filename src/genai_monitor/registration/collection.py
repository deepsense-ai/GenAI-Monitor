import importlib
from typing import Callable

from genai_monitor.db.manager import DBManager
from genai_monitor.injectors.registry import WrapperRegistry
from genai_monitor.registration.utils import DefinitionCollection
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser
from genai_monitor.structures.output_parsers.base import BaseModelOutputParser
from genai_monitor.structures.persistency_manager import PersistencyManager
from genai_monitor.structures.runtime_manager import RuntimeManager


def register_class_collection(
    definition_collection: DefinitionCollection,
    registry: WrapperRegistry,
    db_manager: DBManager,
    persistency_manager: PersistencyManager,
    runtime_manager: RuntimeManager,
    output_parser: BaseModelOutputParser,
    conditioning_parser: BaseConditioningParser,
    hashing_function: Callable,
):
    """
    Register classes defined in ClassDefinitionCollection in GenerativeModelListener.
    Args:
        definition_collection: A collection of definitions of classes to register.
        output_parser: The output parser to use for registration of classes.
        conditioning_parser: The conditioning parser to use for registration of classes.
        hashing_function: The hashing function to register for registration of classes.
        persistency_manager: The persistency manager to use for registration of classes.
        runtime_manager: The runtime manager to use for registration of classes.
        db_manager: The database manager to use for registration of classes.
        registry: The registry to register the classes in.
    """
    for class_definition in definition_collection.data:
        module = importlib.import_module(class_definition.module_name)
        cls = getattr(module, class_definition.cls_name)
        methods_to_wrap = [getattr(cls, method_name) for method_name in class_definition.method_to_wrap]
        registry.register(
            func=methods_to_wrap[0],
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=output_parser,
            conditioning_parser=conditioning_parser,
            hashing_function=hashing_function,
        )


def register_function_collection(
    definition_collection: DefinitionCollection,
    registry: WrapperRegistry,
    db_manager: DBManager,
    persistency_manager: PersistencyManager,
    runtime_manager: RuntimeManager,
    output_parser: BaseModelOutputParser,
    conditioning_parser: BaseConditioningParser,
    hashing_function: Callable,
):
    """
    Register functions defined in FunctionDefinitionCollection.

    Args:
        definition_collection: A collection of definitions of functions to register.
        output_parser: The output parser to use for registration of functions.
        conditioning_parser: The conditioning parser to use for registration of functions.
        hashing_function: The hashing function to register for registration of functions.
        persistency_manager: The persistency manager to use for registration of functions.
        runtime_manager: The runtime manager to use for registration of functions.
        db_manager: The database manager to use for registration of functions.
        registry: The registry to register the functions in.
    """
    for definition in definition_collection.data:
        module = importlib.import_module(definition.module_name)
        func = getattr(module, definition.function_name)
        registry.register(
            func=func,
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=output_parser,
            conditioning_parser=conditioning_parser,
            hashing_function=hashing_function,
        )
