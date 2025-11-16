"""Configuration loader for Hive Mind system."""

import os
import yaml
from typing import Dict, Any

from ..models.model_config import HiveMindConfig, ModelConfig, ModelProvider


def load_config_from_file(config_path: str) -> HiveMindConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return load_config_from_dict(config_data)


def load_config_from_dict(config_data: Dict[str, Any]) -> HiveMindConfig:
    """Load configuration from dictionary."""
    # Process environment variable substitution
    config_data = _substitute_env_vars(config_data)
    
    # Convert model configurations
    models = []
    for model_data in config_data.get("models", []):
        model_config = ModelConfig(**model_data)
        models.append(model_config)
    
    # Create main configuration
    hive_config = HiveMindConfig(
        models=models,
        consensus_algorithm=config_data.get("consensus_algorithm", "weighted_voting"),
        denoising_enabled=config_data.get("denoising_enabled", True),
        max_denoising_iterations=config_data.get("max_denoising_iterations", 3),
        consensus_threshold=config_data.get("consensus_threshold", 0.7),
        reputation_enabled=config_data.get("reputation_enabled", True),
        memory_enabled=config_data.get("memory_enabled", True),
        vector_db_path=config_data.get("vector_db_path", "./chroma_db"),
        redis_url=config_data.get("redis_url", "redis://localhost:6379"),
        max_context_tokens=config_data.get("max_context_tokens", 8000),
        response_timeout_seconds=config_data.get("response_timeout_seconds", 60),
        parallel_requests=config_data.get("parallel_requests", True),
        log_level=config_data.get("log_level", "INFO")
    )
    
    return hive_config


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute environment variables in configuration."""
    if isinstance(obj, dict):
        return {key: _substitute_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        # Extract environment variable name
        env_var = obj[2:-1]
        default_value = None
        
        # Handle default values (e.g., ${VAR:default})
        if ":" in env_var:
            env_var, default_value = env_var.split(":", 1)
        
        # Get environment variable or default
        value = os.getenv(env_var, default_value)
        
        # Convert to appropriate type if possible
        if value is not None:
            # Try to convert to int, float, or keep as string
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        
        return obj
    else:
        return obj


def create_default_config() -> HiveMindConfig:
    """Create a default configuration for testing."""
    models = [
        ModelConfig(
            model_id="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="your-api-key-here",
            max_tokens=2048,
            temperature=0.7,
            enabled=True,
            weight=1.0
        ),
        ModelConfig(
            model_id="claude-instant",
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-instant-1",
            api_key="your-api-key-here",
            max_tokens=2048,
            temperature=0.7,
            enabled=True,
            weight=1.0
        )
    ]
    
    return HiveMindConfig(
        models=models,
        consensus_algorithm="weighted_voting",
        denoising_enabled=True,
        max_denoising_iterations=3,
        consensus_threshold=0.7,
        reputation_enabled=True,
        memory_enabled=True,
        vector_db_path="./chroma_db",
        redis_url="redis://localhost:6379",
        max_context_tokens=8000,
        response_timeout_seconds=60,
        parallel_requests=True,
        log_level="INFO"
    )