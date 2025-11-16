"""Factory for creating model providers."""

from typing import Dict, Type

from .base_model import BaseModelProvider
from .model_config import ModelConfig, ModelProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .cohere_provider import CohereProvider


class ModelFactory:
    """Factory for creating model provider instances."""
    
    _providers: Dict[ModelProvider, Type[BaseModelProvider]] = {
        ModelProvider.OPENAI: OpenAIProvider,
        ModelProvider.ANTHROPIC: AnthropicProvider,
        ModelProvider.COHERE: CohereProvider,
    }
    
    @classmethod
    def create_provider(cls, config: ModelConfig) -> BaseModelProvider:
        """Create a model provider instance from configuration."""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return provider_class(config)
    
    @classmethod
    def register_provider(cls, provider: ModelProvider, provider_class: Type[BaseModelProvider]) -> None:
        """Register a new provider class."""
        cls._providers[provider] = provider_class
    
    @classmethod
    def get_supported_providers(cls) -> list[ModelProvider]:
        """Get list of supported providers."""
        return list(cls._providers.keys())