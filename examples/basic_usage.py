"""Basic usage example for Hive Mind system."""

import asyncio
import os

from src.hive_mind.core.hive_mind import HiveMind
from src.hive_mind.utils.config_loader import create_default_config


async def main():
    """Demonstrate basic Hive Mind usage."""
    print("🧠 Hive Mind - Basic Usage Example")
    print("=" * 50)
    
    # Create a default configuration (you would normally load from config file)
    config = create_default_config()
    
    # Note: You need to set actual API keys in environment variables
    # os.environ["OPENAI_API_KEY"] = "your-openai-key"
    # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
    # os.environ["COHERE_API_KEY"] = "your-cohere-key"
    
    # Initialize Hive Mind
    hive_mind = HiveMind(config)
    
    # Check health of models
    print("\n🔍 Checking model health...")
    health = await hive_mind.health_check()
    for model_id, status in health.items():
        status_icon = "✅" if status["healthy"] else "❌"
        print(f"  {status_icon} {model_id}: {'Healthy' if status['healthy'] else 'Unhealthy'}")
    
    # Process some example inputs
    examples = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the benefits of renewable energy?"
    ]
    
    conversation_id = None
    
    for i, question in enumerate(examples, 1):
        print(f"\n💬 Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Process the input
            result = await hive_mind.process_input(
                user_input=question,
                conversation_id=conversation_id,
                use_memory=True
            )
            
            conversation_id = result["conversation_id"]
            
            # Display results
            print(f"🤖 Response: {result['response']}")
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"🤝 Consensus Strength: {result['consensus_strength']:.2f}")
            print(f"🔀 Divergence: {result['divergence']:.2f}")
            print(f"🔧 Denoising Iterations: {result['denoising_iterations']}")
            print(f"🤖 Contributing Models: {', '.join(result['contributing_models'])}")
            
            # Show individual model responses if desired
            print("\n📋 Individual Model Responses:")
            for resp in result['individual_responses']:
                print(f"  • {resp['model_id']}: {resp['confidence']:.2f} confidence")
                print(f"    {resp['content'][:100]}..." if len(resp['content']) > 100 else f"    {resp['content']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Get system statistics
    print("\n📈 System Statistics:")
    print("-" * 40)
    stats = hive_mind.get_system_stats()
    print(f"Active Conversations: {stats['active_conversations']}")
    print(f"Configured Models: {stats['configured_models']}")
    
    # Get learning insights
    insights = stats['learning_insights']
    print(f"\n🧠 Learning Insights:")
    print(f"Total Conversations in Memory: {insights['conversation_patterns']['total_conversations']}")
    print(f"Average Consensus Confidence: {insights['conversation_patterns']['avg_consensus_confidence']:.2f}")
    
    # Save system state
    print("\n💾 Saving system state...")
    hive_mind.save_state()
    print("✅ State saved successfully!")
    
    print("\n🎉 Example completed!")


if __name__ == "__main__":
    asyncio.run(main())