import os
from typing import Union

from dependency_injector import containers, providers

from genai_monitor.db.config import SessionManager
from genai_monitor.db.manager import DBManager
from genai_monitor.dependencies import DIFFUSERS_AVAILABLE, LITELLM_AVAILABLE, OPENAI_AVAILABLE, TRANSFORMERS_AVAILABLE
from genai_monitor.injectors.registry import ArtifactRegistry, WrapperRegistry
from genai_monitor.injectors.wrapper import ArtifactWrapperFactory, WrapperFactory
from genai_monitor.registration.collection import register_class_collection, register_function_collection
from genai_monitor.registration.utils import ClassDefinitionCollection, FunctionDefinitionCollection
from genai_monitor.structures.persistency_manager import PersistencyManager
from genai_monitor.structures.runtime_manager import RuntimeManager
from genai_monitor.utils.data import get_absolute_path
from genai_monitor.utils.model_hashing import default_model_hashing_function

_TRANSFORMERS_JSON_PATH = "../registration/data/transformers2.json"
_PROVIDERS_JSON_PATH = "../registration/data/providers.json"
_DIFFUSERS_JSON_PATH = "../registration/data/diffusers.json"
_LITELLM_JSON_PATH = "../registration/data/litellm.json"


class DependencyContainer(containers.DeclarativeContainer):
    """Dependency container for the application."""

    config = providers.Configuration()

    wrapper_factory = providers.Singleton(provides=WrapperFactory)
    wrapper_registry = providers.Singleton(provides=WrapperRegistry, wrapper_factory=wrapper_factory)

    # Managers
    session_manager = providers.Factory(provides=SessionManager, database_url=config.db.url)
    db_manager = providers.Factory(provides=DBManager, session_manager=session_manager)
    persistency_manager = providers.Singleton(
        provides=PersistencyManager, path=config.persistency.path, enabled=config.persistency.enabled
    )
    runtime_manager = providers.Singleton(provides=RuntimeManager)

    artifact_wrapper_factory = providers.Singleton(provides=ArtifactWrapperFactory)
    artifact_registry = providers.Singleton(
        provides=ArtifactRegistry, artifact_wrapper_factory=artifact_wrapper_factory
    )

    default_hashing_function = providers.Object(provides=default_model_hashing_function)

    if TRANSFORMERS_AVAILABLE:
        from genai_monitor.structures.conditioning_parsers.transformers_text_generation import (
            TransformersTextGenerationConditioningParser,
        )
        from genai_monitor.structures.output_parsers.transformers_text_generation import (
            TransformersTextGenerationParser,
        )
        from genai_monitor.utils.model_hashing import get_transformers_model_hash

        transformers_output_parser = providers.Factory(provides=TransformersTextGenerationParser)
        transformers_conditioning_parser = providers.Factory(TransformersTextGenerationConditioningParser)
        transformers_hashing_function = providers.Object(provides=get_transformers_model_hash)

        transformers_json_path = providers.Factory(
            get_absolute_path, _TRANSFORMERS_JSON_PATH, relative_to=os.path.abspath(__file__)
        )

        transformers_class_def_collection = providers.Factory(
            ClassDefinitionCollection.from_json, json_path=transformers_json_path
        )

        register_transformers = providers.Callable(
            register_class_collection,
            definition_collection=transformers_class_def_collection,
            registry=wrapper_registry,
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=transformers_output_parser,
            conditioning_parser=transformers_conditioning_parser,
            hashing_function=transformers_hashing_function,
        )

    if OPENAI_AVAILABLE:
        from genai_monitor.structures.conditioning_parsers.openai import OpenAIConditioningParser
        from genai_monitor.structures.output_parsers.openai import OpenAIChatOutputParser
        from genai_monitor.utils.model_hashing import get_empty_model_hash

        openai_output_parser = providers.Factory(provides=OpenAIChatOutputParser)
        openai_conditioning_parser = providers.Factory(OpenAIConditioningParser)
        empty_model_hash = providers.Object(get_empty_model_hash)

        providers_json_path = providers.Factory(
            get_absolute_path, _PROVIDERS_JSON_PATH, relative_to=os.path.abspath(__file__)
        )

        providers_class_def_collection = providers.Factory(
            ClassDefinitionCollection.from_json, json_path=providers_json_path
        )

        register_providers = providers.Callable(
            register_class_collection,
            definition_collection=providers_class_def_collection,
            registry=wrapper_registry,
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=openai_output_parser,
            conditioning_parser=openai_conditioning_parser,
            hashing_function=empty_model_hash,
        )

    if DIFFUSERS_AVAILABLE:
        from genai_monitor.structures.conditioning_parsers.stable_diffusion import StableDiffusionConditioningParser
        from genai_monitor.structures.output_parsers.stable_diffusion import StableDiffusionOutputParser
        from genai_monitor.utils.model_hashing import get_diffusers_model_hash

        diffusers_output_parser = providers.Factory(provides=StableDiffusionOutputParser)
        diffusers_conditioning_parser = providers.Factory(StableDiffusionConditioningParser)
        diffusers_hashing_function = providers.Object(provides=get_diffusers_model_hash)

        diffusers_json_path = providers.Factory(
            get_absolute_path, _DIFFUSERS_JSON_PATH, relative_to=os.path.abspath(__file__)
        )

        diffusers_class_def_collection = providers.Factory(
            ClassDefinitionCollection.from_json, json_path=diffusers_json_path
        )

        register_diffusers = providers.Callable(
            register_class_collection,
            definition_collection=diffusers_class_def_collection,
            registry=wrapper_registry,
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=diffusers_output_parser,
            conditioning_parser=diffusers_conditioning_parser,
            hashing_function=diffusers_hashing_function,
        )

    if LITELLM_AVAILABLE:
        from genai_monitor.structures.conditioning_parsers.litellm import LiteLLMCompletionConditioningParser
        from genai_monitor.structures.output_parsers.litellm import LiteLLMCompletionOutputParser
        from genai_monitor.utils.model_hashing import get_function_full_path

        litellm_output_parser = providers.Factory(provides=LiteLLMCompletionOutputParser)
        litellm_conditioning_parser = providers.Factory(provides=LiteLLMCompletionConditioningParser)
        get_function_full_path = providers.Object(get_function_full_path)

        litellm_json_path = providers.Factory(
            get_absolute_path, _LITELLM_JSON_PATH, relative_to=os.path.abspath(__file__)
        )

        litellm_def_collection = providers.Factory(FunctionDefinitionCollection.from_json, json_path=litellm_json_path)

        register_litellm = providers.Callable(
            register_function_collection,
            definition_collection=litellm_def_collection,
            registry=wrapper_registry,
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=litellm_output_parser,
            conditioning_parser=litellm_conditioning_parser,
            hashing_function=get_function_full_path,
        )


_container: Union[DependencyContainer, None] = None


def get_container() -> DependencyContainer:
    """Get the dependency container.

    Returns:
        The dependency container.
    """
    global _container  # noqa: PLW0603
    if _container is None:
        _container = DependencyContainer()
    return _container


def reset_container() -> None:
    """Reset the container to None."""
    global _container  # noqa: PLW0603
    _container = None
