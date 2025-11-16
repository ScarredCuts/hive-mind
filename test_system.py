#!/usr/bin/env python3
"""Quick test script to verify Hive Mind installation."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hive_mind.utils.config_loader import create_default_config
from hive_mind.core.hive_mind import HiveMind


async def test_basic_functionality():
    """Test basic Hive Mind functionality without API calls."""
    print("🧠 Testing Hive Mind System")
    print("=" * 50)
    
    try:
        # Create configuration
        print("📋 Creating configuration...")
        config = create_default_config()
        print(f"✅ Configuration created with {len(config.models)} models")
        
        # Initialize Hive Mind
        print("\n🚀 Initializing Hive Mind...")
        hive_mind = HiveMind(config)
        print("✅ Hive Mind initialized successfully")
        
        # Test consensus engine
        print("\n🤝 Testing consensus engine...")
        from hive_mind.models.response import ModelResponse, ModelProvider
        
        # Create mock responses
        responses = [
            ModelResponse(
                model_id="test1",
                provider=ModelProvider.OPENAI,
                content="The capital of France is Paris.",
                confidence=0.9,
                response_time_ms=1000,
                token_usage={"total_tokens": 50}
            ),
            ModelResponse(
                model_id="test2", 
                provider=ModelProvider.ANTHROPIC,
                content="Paris is the capital city of France.",
                confidence=0.85,
                response_time_ms=1200,
                token_usage={"total_tokens": 45}
            )
        ]
        
        result = await hive_mind.consensus_engine.synthesize(responses)
        print(f"✅ Consensus generated: {result.content[:50]}...")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Consensus Strength: {result.consensus_strength:.3f}")
        
        # Test denoising engine
        print("\n🔧 Testing denoising engine...")
        denoised_responses, iterations = await hive_mind.denoising_engine.denoise_responses(responses)
        print(f"✅ Denoising completed in {iterations} iterations")
        print(f"   {len(denoised_responses)} responses processed")
        
        # Test reputation system
        print("\n⭐ Testing reputation system...")
        for response in responses:
            hive_mind.reputation_manager.update_reputation(response)
        
        stats = hive_mind.reputation_manager.get_all_stats()
        print(f"✅ Reputation updated for {len(stats)} models")
        
        # Test memory system
        print("\n🧠 Testing memory system...")
        from hive_mind.models.response import ConsensusResponse
        
        consensus_response = ConsensusResponse(
            content="Test consensus response",
            contributing_responses=["test1", "test2"],
            confidence_score=0.88,
            consensus_strength=0.92,
            divergence_score=0.15,
            synthesis_method="weighted_voting"
        )
        
        hive_mind.memory_manager.store_conversation(
            "test-conv", "What is the capital of France?", 
            consensus_response, responses
        )
        print("✅ Conversation stored in memory")
        
        # Get system stats
        print("\n📊 System Statistics:")
        stats = hive_mind.get_system_stats()
        print(f"   Active Conversations: {stats['active_conversations']}")
        print(f"   Configured Models: {stats['configured_models']}")
        print(f"   Consensus Algorithm: {stats['config']['consensus_algorithm']}")
        print(f"   Denoising Enabled: {stats['config']['denoising_enabled']}")
        print(f"   Reputation Enabled: {stats['config']['reputation_enabled']}")
        print(f"   Memory Enabled: {stats['config']['memory_enabled']}")
        
        print("\n🎉 All tests passed! Hive Mind is working correctly.")
        print("\n📝 Next Steps:")
        print("   1. Set your API keys in environment variables")
        print("   2. Run: python main.py")
        print("   3. Open: http://localhost:8000/web/index.html")
        print("   4. Try: python examples/basic_usage.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)