#!/usr/bin/env python3
"""Simple test to verify core structure without heavy dependencies."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports work."""
    print("🧠 Testing Hive Mind Imports")
    print("=" * 40)
    
    try:
        # Test basic model imports
        print("📦 Testing model imports...")
        from hive_mind.models.response import ModelResponse, ConsensusResponse
        from hive_mind.models.model_config import ModelConfig, HiveMindConfig
        print("✅ Model imports successful")
        
        # Test configuration
        print("\n⚙️ Testing configuration...")
        from hive_mind.utils.config_loader import create_default_config
        config = create_default_config()
        print(f"✅ Configuration created with {len(config.models)} models")
        
        # Test basic model creation
        print("\n🤖 Testing model creation...")
        
        # Test that we can create config objects
        model_config = ModelConfig(
            model_id="test",
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-key"
        )
        print("✅ Model configuration created")
        
        # Test response creation
        response = ModelResponse(
            model_id="test",
            provider="openai",
            content="Test response",
            confidence=0.8,
            response_time_ms=1000,
            token_usage={"total_tokens": 50}
        )
        print("✅ Model response created")
        
        # Test consensus response
        consensus = ConsensusResponse(
            content="Test consensus",
            contributing_responses=["test"],
            confidence_score=0.85,
            consensus_strength=0.9,
            divergence_score=0.1,
            synthesis_method="test"
        )
        print("✅ Consensus response created")
        
        print("\n🎉 All basic tests passed!")
        print("\n📝 System Structure Verified:")
        print("   ✅ Data models working")
        print("   ✅ Configuration system working")
        print("   ✅ Basic object creation working")
        
        print("\n🚀 Next Steps:")
        print("   1. Install full dependencies: pip install -r requirements.txt")
        print("   2. Set API keys in environment variables")
        print("   3. Run: python main.py")
        print("   4. Open: http://localhost:8000/web/index.html")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)