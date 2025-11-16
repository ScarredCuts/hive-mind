"""Base interface for AI model providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .response import ModelResponse
from .model_config import ModelConfig


class BaseModelProvider(ABC):
    """Abstract base class for AI model providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.model_id
        self.provider = config.provider
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model provider is healthy and accessible."""
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Estimate token count for given text."""
        pass
    
    def prepare_prompt(
        self,
        user_input: str,
        context: Optional[List[str]] = None
    ) -> str:
        """Prepare prompt with context."""
        if not context:
            return user_input
        
        context_str = "\n".join(context)
        return f"Context:\n{context_str}\n\nUser: {user_input}\nAssistant:"