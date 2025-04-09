import importlib
import inspect
import json
import os
import re
import sys
from dataclasses import fields, is_dataclass
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from attrs import asdict, define
from cattrs import structure
from loguru import logger

T = TypeVar("T")


def _accepts_self(func: Callable) -> bool:
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    return (parameters and parameters[0].name == "self")

def _make_bound_method(f: Callable) -> Callable:
    @wraps(f)
    def _wrapper(self, *args, **kwargs): # noqa: ANN001,ANN002, ANN003
        return f(*args, **kwargs)

    return _wrapper


def _make_cls(cls_name: str, base: Optional[Type[T]] = None, method_mapper: Dict[str, Callable] = None) -> Type[T]:
    method_mapper = method_mapper or {}
    bound_method_mapper = {}
    for method_name, method in method_mapper.items():
        if _accepts_self(method):
            bound_method_mapper[method_name] = method
        else:
            bound_method_mapper[method_name] = _make_bound_method(method)
    if base is None:
        base = object
    cls = type(cls_name, (base,), bound_method_mapper)
    return cast(Type[T], cls)


@define
class ClassDefinition:
    """Data Transfer Object representing a definition of a class to observe.

    Attributes:
        cls_name: The name of the class.
        module_name: The name of the module containing the class.
        method_to_wrap: The method to wrap within the class.
    """
    cls_name: str
    module_name: str
    method_to_wrap: List[str]

    @classmethod
    def from_full_name(cls, full_name: str, method_to_wrap: Union[str, List[str]]) -> "ClassDefinition":
        """Create a `ClassDefinitionDTO` instance from a fully qualified class name.

        Args:
            full_name: The fully qualified name of the class (e.g., "module.class_name").
            method_to_wrap: The method to wrap within the class.

        Returns:
            An instance of `ClassDefinitionDTO` populated with the parsed class and module names.
        """
        full_name_parts = full_name.split(".")
        module_name, cls_name = ".".join(full_name_parts[:-1]), full_name_parts[-1]
        if isinstance(method_to_wrap, str):
            method_to_wrap = [method_to_wrap]
        return cls(cls_name=cls_name, module_name=module_name, method_to_wrap=method_to_wrap)


@define
class FunctionDefinition:
    """Data Transfer Object representing a definition of a function to observe.

    Attributes:
        function_name: The name of the function.
        module_name: The name of the module containing the function.
    """
    function_name: str
    module_name: str

    @classmethod
    def from_full_name(cls, full_name: str) -> "FunctionDefinition":
        """Create a `FunctionDefinitionDTO` instance from a fully qualified function name.

        Args:
            full_name: The fully qualified name of the function (e.g., "module.function_name").

        Returns:
            An instance of `FunctionDefinitionDTO` populated with the parsed function and module names.
        """
        module_name, function_name = full_name.rsplit(".", 1)
        return cls(function_name=function_name, module_name=module_name)


@define
class DefinitionCollection:
    """A collection of definitions to observe."""
    data: Union[List[ClassDefinition], List[FunctionDefinition]]

    def merge(self, other: "DefinitionCollection"):
        """Merge another `DefinitionCollection` into this collection.

        Args:
            other: Another `DefinitionCollection` instance to merge.
        """
        self.data.extend(other.data)

    def to_json(self, save_path: str):
        """Save the collection as a JSON file.

        Args:
            save_path: The file path to save the JSON representation of the collection.
        """
        with open(save_path, "w") as f:
            json.dump(asdict(self), f, indent=4)


@define
class ClassDefinitionCollection(DefinitionCollection):
    """A collection of class definitions."""
    data: List[ClassDefinition]

    @classmethod
    def from_json(cls, json_path: str) -> "ClassDefinitionCollection":
        """Load ClassDefinitionCollection from json.

        Args:
            json_path: The path to the json file containing serialized ClassDefinitionCollection.

        Returns:
            ClassDefinitionCollection: a class instance populated with data from json.

        """
        with open(json_path) as f:
            data = json.load(f)
        return structure(data, cls)


@define
class FunctionDefinitionCollection(DefinitionCollection):
    """A collection of function definitions."""
    data: List[FunctionDefinition]

    @classmethod
    def from_json(cls, json_path: str) -> "FunctionDefinitionCollection":
        """Load FunctionDefinitionCollection from json.

        Args:
            json_path: The path to the json file containing serialized FunctionDefinitionCollection.

        Returns:
            FunctionDefinitionCollection: a class instance populated with data from json.
        """
        with open(json_path) as f:
            data = json.load(f)
        return structure(data, cls)


def get_full_name_from_class(cls: Type) -> str:
    """Get the fully qualified name of a class.

    Args:
        cls: The class to retrieve the name for.

    Returns:
        The fully qualified name of the class in the format `module.submodules.class_name`.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def is_subclass_predicate(checked_cls: Type, base_cls: Type) -> bool:
    """Check if a class is a subclass of another class.

    Args:
        checked_cls: The class to check.
        base_cls: The base class to check against.

    Returns:
        True if the checked class is a subclass of the base
    """
    return inspect.isclass(checked_cls) and issubclass(checked_cls, base_cls)


def get_module_classes(module: ModuleType, predicate: Callable = inspect.isclass) -> List[Tuple[str, Type]]:
    """Get all classes from a module that satisfy a predicate.

    Args:
        module: The module to get classes from.
        predicate: The predicate to filter classes by.

    Returns:
        A list of tuples containing the name and class of each class in the module
        that satisfies the predicate.
    """
    return inspect.getmembers(sys.modules[module.__name__], predicate=predicate)


def get_method_return_type_from_docstring(method: Any) -> Optional[str]:
    """Extract the return type from a method's docstring.

    Args:
        method: The method to extract the return type from.

    Returns:
        The return type of the method or None if it cannot be extracted.
    """
    try:
        docstring = inspect.getdoc(method)
        if not docstring:
            return None

        # This regular expression is tailored for diffusers documentation and untested on other libraries and formats.
        match = re.search(r"Returns:\s*\[?`?([^\s\[`]+)[`|\]]?", docstring, re.IGNORECASE)

        if match:
            return match.group(1)

    except Exception as e:
        print(f"Error while extracting return type: {e}")

    return None


def get_class_fields(cls: Type) -> Union[List[str], None]:
    """Get the fields of a class.

    Args:
        cls: The class to get fields for.

    Returns:
        The fields of the class or None if the class is not a dataclass or regular class.
    """
    if is_dataclass(cls):
        return [field.name for field in fields(cls)]

    if inspect.isclass(cls):
        members = inspect.getmembers(cls, lambda member: not inspect.isroutine(member))
        fields_dict = {
            name: type(value).__name__ if value is not None else "Unknown"
            for name, value in members
            if not name.startswith("__")
        }
        return fields_dict if fields_dict else None

    return None


def find_class_in_framework_via_fs(base_package: str, target_class: str) -> Optional[Type]:
    """Find a class in a framework by searching the filesystem.

    Args:
        base_package: The base package to search in.
        target_class: The class to search for.

    Returns:
        The class if found, otherwise None.
    """
    try:
        base_module = importlib.import_module(base_package)
        base_path = os.path.dirname(base_module.__file__)
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    module_rel_path = os.path.relpath(root, base_path)
                    module_name = os.path.splitext(file)[0]
                    if module_rel_path == ".":
                        full_module_path = f"{base_package}.{module_name}"
                    else:
                        rel_module = module_rel_path.replace(os.path.sep, ".")
                        full_module_path = f"{base_package}.{rel_module}.{module_name}"
                    try:
                        module = importlib.import_module(full_module_path)

                        if hasattr(module, target_class):
                            return getattr(module, target_class)
                    except ImportError:
                        continue
    except ImportError:
        logger.info(f"Failed to import base package {base_package}")

    return None
