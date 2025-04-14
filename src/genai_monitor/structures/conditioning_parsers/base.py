import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Set, Tuple

from genai_monitor.common.errors import NotJsonableError
from genai_monitor.common.structures.data import Conditioning, Sample
from genai_monitor.common.utils import is_jsonable
from genai_monitor.db.manager import DBManager
from genai_monitor.dependencies import DIFFUSERS_AVAILABLE, OPENAI_AVAILABLE, TRANSFORMERS_AVAILABLE
from genai_monitor.static.fields import CONDITIONING_METADATA_FIELDNAME
from genai_monitor.structures.persistency_manager import PersistencyManager
from genai_monitor.utils.data_hashing import Jsonable, get_hash_from_jsonable

from .seed_types import SeedType


class BaseConditioningParser(ABC):
    """Base class for conditioning parsers.

    Use it to create custom conditioning parsers, not available natively in the library.

    'parse_func_arguments' is the core method that needs to be overwritten by the subclasses.
    """

    _tracked_seed_types: Set[SeedType] = set()  # Override in subclasses to specify which seeds to track
    db_manager: DBManager
    persistency_manager: PersistencyManager

    def parse_conditioning(self, method: Callable, *args, **kwargs) -> Tuple[Conditioning, List[Sample]]:
        """Parse the execution parameters of a function into a Conditioning object.

        Inspects the signature of the function and passed arguments/keyword arguments.

        Args:
            method: The method
            *args: Arguments of the method.
            **kwargs: Keyword arguments of the method.

        Returns:
            A Conditioning object parsed constructed based on the execution of the method.

        Raises:
            NotJsonableError: when the parsed arguments cannot be serialized to json.
        """
        conditioning_metadata = kwargs.pop(CONDITIONING_METADATA_FIELDNAME, None)
        inference_params = self._get_call_params_with_defaults(method, *args, **kwargs)
        # Get the instance from the self parameter in wrapped_inference_method
        instance = kwargs.get("self")

        if instance is not None:
            inference_params["__instance__"] = instance

        jsonable_value = self.parse_func_arguments(**inference_params)
        if not is_jsonable(jsonable_value):
            raise NotJsonableError(jsonable_value)

        # Extract seeds and add to metadata
        seed_metadata = self._extract_seeds(**inference_params)

        if conditioning_metadata:
            if not is_jsonable(conditioning_metadata):
                raise NotJsonableError(
                    f"Conditioning metadata not jsonable: {CONDITIONING_METADATA_FIELDNAME} = {conditioning_metadata}"
                )
            if isinstance(jsonable_value, dict):
                jsonable_value[CONDITIONING_METADATA_FIELDNAME] = conditioning_metadata
            else:
                jsonable_value = {"content": jsonable_value, "metadata": conditioning_metadata}  # type: ignore

        conditioning_hash = get_hash_from_jsonable(jsonable_value)

        return Conditioning(
            value=jsonable_value,
            hash=conditioning_hash,
            value_metadata={
                "seeds": seed_metadata if seed_metadata else None,
            },
        )

    def _extract_seeds(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Extract seeds from the inference parameters based on tracked seed types.

        Returns:
            Dictionary mapping seed type to seed value, or None if no seeds found
        """
        seeds = {}

        for seed_type in self._tracked_seed_types:
            if (DIFFUSERS_AVAILABLE or TRANSFORMERS_AVAILABLE) and seed_type == SeedType.TORCH:
                import torch

                for _, v in kwargs.items():
                    if isinstance(v, torch.Generator):
                        seeds[seed_type.name] = sum(v.get_state().tolist())
            elif OPENAI_AVAILABLE and seed_type == SeedType.OPENAI:
                from openai import NotGiven

                if "seed" in kwargs:
                    seeds[seed_type.name] = kwargs["seed"]
                    if isinstance(seeds[seed_type.name], NotGiven):
                        seeds[seed_type.name] = None
            # Add more seed type handling as needed

        return seeds if seeds else None

    @abstractmethod
    def parse_func_arguments(self, *args, **kwargs) -> Jsonable:
        """Core function to be overwritten in subclasses.

        Parse func arguments and convert into a jsonable object - the parsing and conversion approach may vary depending
        on the type of func arguments.

        Args:
            *args: Arguments of the method.
            **kwargs: Keyword arguments of the method.

        Returns:
            Parsed parameters that can be serialized to json.
        """

    @staticmethod
    def _get_call_params_with_defaults(func: Callable, *args, **kwargs) -> OrderedDict[str, Any]:
        sig = inspect.signature(func)
        if "self" in sig.parameters:
            bound_args = sig.bind_partial(*(None, *args), **kwargs)
        else:
            bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        raw_bound_arguments = bound_args.arguments

        # Remove self argument from the dictionary
        raw_bound_arguments.pop("self", None)
        return raw_bound_arguments


class DefaultConditioningParser(BaseConditioningParser):
    """Default parser that creates a Conditioning object from all json convertible parameters of an inference."""

    def parse_func_arguments(self, *args, **kwargs) -> Jsonable:
        """Ensures any non-JSON-serializable values are converted to string, thereby bypassing the NotJsonableError.

        Parameters:
            *args: Arguments of the method.
            **kwargs: Keyword arguments of the method.

        Returns:
            A dictionary containing the parsed arguments.
        """
        parsed_arguments = deepcopy(kwargs)

        for param, val in parsed_arguments.items():
            if not is_jsonable(val):
                parsed_arguments[param] = str(val)

        return {param: val for param, val in parsed_arguments.items() if is_jsonable(val)}  # type: ignore
