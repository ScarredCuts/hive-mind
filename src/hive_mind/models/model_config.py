"""Configuration models for AI models."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .response import ModelProvider


class ModelConfig(BaseModel):
    """Configuration for a single AI model."""
    
    model_id: str
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = Field(default=2048, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, gt=0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    timeout_seconds: int = Field(default=30, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0)
    cost_per_token: Optional[float] = Field(None, ge=0.0)
    
    class Config:
        extra = "allow"


class HiveMindConfig(BaseModel):
    """Main configuration for the Hive Mind system."""
    
    models: List[ModelConfig]
    consensus_algorithm: str = Field(default="weighted_voting")
    denoising_enabled: bool = True
    max_denoising_iterations: int = Field(default=3, ge=0)
    consensus_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    reputation_enabled: bool = True
    memory_enabled: bool = True
    vector_db_path: str = "./chroma_db"
    redis_url: str = "redis://localhost:6379"
    max_context_tokens: int = Field(default=8000, gt=0)
    response_timeout_seconds: int = Field(default=60, gt=0)
    parallel_requests: bool = True
    log_level: str = "INFO"
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models."""
        return [model for model in self.models if model.enabled]
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None