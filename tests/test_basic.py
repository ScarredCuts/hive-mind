"""Basic tests for Hive Mind system."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.hive_mind.models.response import ModelResponse, ModelProvider
from src.hive_mind.models.model_config import ModelConfig, HiveMindConfig
from src.hive_mind.reputation.reputation_manager import ReputationManager
from src.hive_mind.consensus.consensus_engine import ConsensusEngine
from src.hive_mind.denoising.denoising_engine import DenoisingEngine


class TestReputationManager:
    """Test the reputation management system."""
    
    def test_reputation_entry_creation(self):
        """Test creating a reputation entry."""
        from src.hive_mind.reputation.reputation_manager import ReputationEntry
        
        entry = ReputationEntry("test-model")
        assert entry.model_id == "test-model"
        assert entry.total_responses == 0
        assert entry.weight == 1.0
    
    def test_reputation_update(self):
        """Test updating reputation with a response."""
        from src.hive_mind.reputation.reputation_manager import ReputationEntry
        
        entry = ReputationEntry("test-model")
        
        response = ModelResponse(
            model_id="test-model",
            provider=ModelProvider.OPENAI,
            content="Test response",
            confidence=0.8,
            response_time_ms=1000,
            token_usage={"total_tokens": 100}
        )
        
        entry.update(response)
        
        assert entry.total_responses == 1
        assert entry.successful_responses == 1
        assert len(entry.response_times) == 1
        assert entry.response_times[0] == 1000


class TestConsensusEngine:
    """Test the consensus engine."""
    
    @pytest.fixture
    def consensus_engine(self):
        """Create a consensus engine for testing."""
        return ConsensusEngine()
    
    @pytest.fixture
    def sample_responses(self):
        """Create sample model responses for testing."""
        return [
            ModelResponse(
                model_id="model1",
                provider=ModelProvider.OPENAI,
                content="The capital of France is Paris.",
                confidence=0.9,
                response_time_ms=1000,
                token_usage={"total_tokens": 50}
            ),
            ModelResponse(
                model_id="model2",
                provider=ModelProvider.ANTHROPIC,
                content="Paris is the capital city of France.",
                confidence=0.85,
                response_time_ms=1200,
                token_usage={"total_tokens": 45}
            ),
            ModelResponse(
                model_id="model3",
                provider=ModelProvider.COHERE,
                content="France's capital is Paris.",
                confidence=0.8,
                response_time_ms=800,
                token_usage={"total_tokens": 40}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_weighted_voting_consensus(self, consensus_engine, sample_responses):
        """Test weighted voting consensus algorithm."""
        result = await consensus_engine.synthesize(
            sample_responses,
            algorithm="weighted_voting"
        )
        
        assert result is not None
        assert result.content is not None
        assert result.confidence_score > 0
        assert result.consensus_strength > 0
        assert len(result.contributing_responses) > 0
        assert result.synthesis_method == "weighted_voting"
    
    @pytest.mark.asyncio
    async def test_majority_voting_consensus(self, consensus_engine, sample_responses):
        """Test majority voting consensus algorithm."""
        result = await consensus_engine.synthesize(
            sample_responses,
            algorithm="majority_voting"
        )
        
        assert result is not None
        assert result.content is not None
        assert result.synthesis_method == "majority_voting"


class TestDenoisingEngine:
    """Test the denoising engine."""
    
    @pytest.fixture
    def denoising_engine(self):
        """Create a denoising engine for testing."""
        return DenoisingEngine()
    
    def test_noise_detection(self, denoising_engine):
        """Test noise detection capabilities."""
        detector = denoising_engine.noise_detector
        
        # Test hallucination detection
        hallucination_content = "As an AI language model, I don't have access to current information."
        hallucination_score = detector.detect_hallucination(hallucination_content)
        assert hallucination_score > 0.5
        
        # Test repetition detection
        repetitive_content = "This is repeated. This is repeated. This is repeated."
        repetition_score = detector.detect_repetition(repetitive_content)
        assert repetition_score > 0.3
        
        # Test good content
        good_content = "The capital of France is Paris, a beautiful city known for the Eiffel Tower."
        hallucination_score = detector.detect_hallucination(good_content)
        repetition_score = detector.detect_repetition(good_content)
        assert hallucination_score < 0.3
        assert repetition_score < 0.3
    
    @pytest.mark.asyncio
    async def test_response_denoising(self, denoising_engine):
        """Test response denoising process."""
        # Create responses with varying quality
        responses = [
            ModelResponse(
                model_id="model1",
                provider=ModelProvider.OPENAI,
                content="As an AI language model, I cannot provide specific information.",
                confidence=0.3,
                response_time_ms=1000,
                token_usage={"total_tokens": 100}
            ),
            ModelResponse(
                model_id="model2",
                provider=ModelProvider.ANTHROPIC,
                content="The capital of France is Paris, known for the Eiffel Tower and Louvre Museum.",
                confidence=0.9,
                response_time_ms=800,
                token_usage={"total_tokens": 80}
            )
        ]
        
        denoised_responses, iterations = await denoising_engine.denoise_responses(responses)
        
        assert len(denoised_responses) == 2
        assert iterations >= 0
        
        # Check that quality scores were assigned
        for response in denoised_responses:
            assert response.quality_score is not None
            assert response.quality_rating is not None


class TestModelFactory:
    """Test the model factory."""
    
    def test_provider_creation(self):
        """Test creating model providers."""
        from src.hive_mind.models.model_factory import ModelFactory
        
        # Test OpenAI provider creation
        openai_config = ModelConfig(
            model_id="test-openai",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        provider = ModelFactory.create_provider(openai_config)
        assert provider is not None
        assert provider.model_id == "test-openai"
        assert provider.provider == ModelProvider.OPENAI
        
        # Test Anthropic provider creation
        anthropic_config = ModelConfig(
            model_id="test-anthropic",
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-instant-1",
            api_key="test-key"
        )
        
        provider = ModelFactory.create_provider(anthropic_config)
        assert provider is not None
        assert provider.model_id == "test-anthropic"
        assert provider.provider == ModelProvider.ANTHROPIC


class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid configuration
        config = ModelConfig(
            model_id="test-model",
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            max_tokens=2048,
            temperature=0.7,
            enabled=True,
            weight=1.0
        )
        
        assert config.model_id == "test-model"
        assert config.provider == ModelProvider.OPENAI
        assert config.temperature == 0.7
        assert config.enabled is True
        
        # Test invalid temperature (should be validated by Pydantic)
        with pytest.raises(ValueError):
            ModelConfig(
                model_id="test-model",
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                api_key="test-key",
                temperature=3.0  # Invalid: should be <= 2.0
            )
    
    def test_hive_config(self):
        """Test Hive Mind configuration."""
        models = [
            ModelConfig(
                model_id="model1",
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                api_key="test-key",
                enabled=True
            ),
            ModelConfig(
                model_id="model2",
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-instant-1",
                api_key="test-key",
                enabled=False
            )
        ]
        
        config = HiveMindConfig(
            models=models,
            consensus_algorithm="weighted_voting",
            denoising_enabled=True,
            reputation_enabled=True,
            memory_enabled=True
        )
        
        # Test getting enabled models
        enabled_models = config.get_enabled_models()
        assert len(enabled_models) == 1
        assert enabled_models[0].model_id == "model1"
        
        # Test getting model by ID
        model = config.get_model_by_id("model1")
        assert model is not None
        assert model.model_id == "model1"
        
        model = config.get_model_by_id("nonexistent")
        assert model is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])