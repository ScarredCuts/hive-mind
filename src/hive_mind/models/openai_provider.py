"""OpenAI model provider implementation."""

import time
from typing import Dict, List, Optional

import openai
from openai import AsyncOpenAI

from .base_model import BaseModelProvider
from .response import ModelResponse
from .model_config import ModelConfig


class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout_seconds
        )
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, context)
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                **kwargs
            )
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract content and metadata
            content = response.choices[0].message.content or ""
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            # Estimate confidence based on token usage and response characteristics
            confidence = self._estimate_confidence(content, token_usage)
            
            return ModelResponse(
                model_id=self.model_id,
                provider=self.provider,
                content=content,
                confidence=confidence,
                response_time_ms=response_time_ms,
                token_usage=token_usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                }
            )
            
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return ModelResponse(
                model_id=self.model_id,
                provider=self.provider,
                content=f"Error: {str(e)}",
                confidence=0.0,
                response_time_ms=response_time_ms,
                token_usage={},
                metadata={"error": str(e)}
            )
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.model_name)
            return len(encoding.encode(text))
        except Exception:
            # Fallback estimation (roughly 4 chars per token)
            return len(text) // 4
    
    def _prepare_messages(self, prompt: str, context: Optional[List[str]]) -> List[Dict]:
        """Prepare messages for OpenAI chat API."""
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
        if context:
            for ctx in context:
                messages.append({"role": "user", "content": ctx})
        
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def _estimate_confidence(self, content: str, token_usage: Dict) -> float:
        """Estimate confidence based on response characteristics."""
        # Base confidence
        confidence = 0.8
        
        # Adjust based on content length
        if len(content) < 10:
            confidence -= 0.3
        elif len(content) > 1000:
            confidence += 0.1
        
        # Adjust based on token usage
        total_tokens = token_usage.get("total_tokens", 0)
        if total_tokens > 0:
            completion_ratio = token_usage.get("completion_tokens", 0) / total_tokens
            if completion_ratio > 0.8:
                confidence -= 0.1  # Might be cut off
        
        # Check for error indicators
        if content.startswith("Error:") or "I'm sorry" in content:
            confidence -= 0.4
        
        return max(0.0, min(1.0, confidence))