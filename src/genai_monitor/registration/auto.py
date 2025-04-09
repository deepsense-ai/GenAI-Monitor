from loguru import logger

from genai_monitor.dependencies import DIFFUSERS_AVAILABLE, LITELLM_AVAILABLE, OPENAI_AVAILABLE, TRANSFORMERS_AVAILABLE
from genai_monitor.injectors.containers import get_container

container = get_container()

if TRANSFORMERS_AVAILABLE:
    container.register_transformers()
    logger.info("Automatically registered transformers classes.")

if DIFFUSERS_AVAILABLE:
    container.register_diffusers()
    logger.info("Automatically registered diffusers classes.")

if LITELLM_AVAILABLE:
    container.register_litellm()
    logger.info("Automatically registered LiteLLM completion.")

if OPENAI_AVAILABLE:
    if LITELLM_AVAILABLE:
        logger.warning("OpenAI provider is not available when LiteLLM is enabled.")
    else:
        container.register_providers()
        logger.info("Automatically registered OpenAI classes.")
