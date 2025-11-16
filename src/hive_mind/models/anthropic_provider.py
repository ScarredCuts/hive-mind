"""Anthropic Claude model provider implementation."""

import time
from typing import Dict, List, Optional

import anthropic

from .base_model import BaseModelProvider
from .response import ModelResponse
from .model_config import ModelConfig


class AnthropicProvider(BaseModelProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(
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
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        try:
            # Prepare prompt with context
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Make API call
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": full_prompt}],
                **kwargs
            )
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract content and metadata
            content = response.content[0].text if response.content else ""
            token_usage = {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
            }
            
            # Estimate confidence
            confidence = self._estimate_confidence(content, token_usage)
            
            return ModelResponse(
                model_id=self.model_id,
                provider=self.provider,
                content=content,
                confidence=confidence,
                response_time_ms=response_time_ms,
                token_usage=token_usage,
                metadata={
                    "stop_reason": response.stop_reason,
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
        """Check Anthropic API health."""
        try:
            await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception:
            return False
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count for Anthropic models."""
        # Anthropic uses a different tokenizer, but we can estimate
        # Claude models typically use ~4 characters per token
        return len(text) // 4
    
    def _prepare_prompt(self, prompt: str, context: Optional[List[str]]) -> str:
        """Prepare prompt with context for Claude."""
        if not context:
            return prompt
        
        context_str = "\n".join(context)
        return f"""Context:
{context_str}

User: {prompt}

Please provide a helpful response based on the context above."""
    
    def _estimate_confidence(self, content: str, token_usage: Dict) -> float:
        """Estimate confidence based on response characteristics."""
        confidence = 0.8
        
        # Adjust based on content length
        if len(content) < 10:
            confidence -= 0.3
        elif len(content) > 1000:
            confidence += 0.1
        
        # Adjust based on token usage
        total_tokens = token_usage.get("total_tokens", 0)
        if total_tokens > 0:
            output_ratio = token_usage.get("output_tokens", 0) / total_tokens
            if output_ratio > 0.8:
                confidence -= 0.1
        
        # Check for error indicators
        if content.startswith("Error:") or "apologize" in content.lower():
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))