"""Example of comparing consensus algorithms."""

import asyncio
import time

from src.hive_mind.core.hive_mind import HiveMind
from src.hive_mind.utils.config_loader import create_default_config


async def compare_algorithms():
    """Compare different consensus algorithms."""
    print("🔬 Hive Mind - Consensus Algorithm Comparison")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config()
    
    # Test questions
    test_questions = [
        "What are the main causes of climate change?",
        "How does photosynthesis work?",
        "Explain the concept of blockchain technology.",
        "What is the meaning of life?"
    ]
    
    algorithms = ["weighted_voting", "majority_voting"]
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🧪 Testing algorithm: {algorithm}")
        print("-" * 40)
        
        # Create new Hive Mind instance with specific algorithm
        config.consensus_algorithm = algorithm
        hive_mind = HiveMind(config)
        
        algorithm_results = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ Question {i}: {question}")
            
            start_time = time.time()
            
            try:
                result = await hive_mind.process_input(
                    user_input=question,
                    use_memory=False  # Disable memory for fair comparison
                )
                
                processing_time = time.time() - start_time
                
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                print(f"📊 Confidence: {result['confidence']:.3f}")
                print(f"🤝 Consensus Strength: {result['consensus_strength']:.3f}")
                print(f"🔀 Divergence: {result['divergence']:.3f}")
                print(f"🤖 Models used: {len(result['contributing_models'])}")
                
                algorithm_results.append({
                    'question': question,
                    'response': result['response'],
                    'confidence': result['confidence'],
                    'consensus_strength': result['consensus_strength'],
                    'divergence': result['divergence'],
                    'processing_time': processing_time,
                    'models_used': len(result['contributing_models']),
                    'denoising_iterations': result['denoising_iterations']
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                algorithm_results.append({
                    'question': question,
                    'error': str(e)
                })
        
        results[algorithm] = algorithm_results
    
    # Compare results
    print("\n📊 Algorithm Comparison Summary")
    print("=" * 60)
    
    for algorithm, algo_results in results.items():
        print(f"\n🔹 {algorithm.upper()}:")
        
        # Calculate averages (excluding errors)
        valid_results = [r for r in algo_results if 'error' not in r]
        
        if valid_results:
            avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
            avg_consensus = sum(r['consensus_strength'] for r in valid_results) / len(valid_results)
            avg_divergence = sum(r['divergence'] for r in valid_results) / len(valid_results)
            avg_time = sum(r['processing_time'] for r in valid_results) / len(valid_results)
            
            print(f"  Average Confidence: {avg_confidence:.3f}")
            print(f"  Average Consensus Strength: {avg_consensus:.3f}")
            print(f"  Average Divergence: {avg_divergence:.3f}")
            print(f"  Average Processing Time: {avg_time:.3f}s")
            print(f"  Success Rate: {len(valid_results)}/{len(algo_results)} ({len(valid_results)/len(algo_results)*100:.1f}%)")
        else:
            print("  ❌ All requests failed")
    
    # Show side-by-side comparison for each question
    print("\n🔍 Side-by-Side Comparison:")
    print("=" * 60)
    
    for i, question in enumerate(test_questions):
        print(f"\n❓ Question {i+1}: {question}")
        print("-" * 40)
        
        for algorithm in algorithms:
            result = results[algorithm][i]
            if 'error' not in result:
                print(f"\n🔹 {algorithm.upper()}:")
                print(f"  📊 Confidence: {result['confidence']:.3f}")
                print(f"  🤝 Consensus: {result['consensus_strength']:.3f}")
                print(f"  🔀 Divergence: {result['divergence']:.3f}")
                print(f"  ⏱️  Time: {result['processing_time']:.3f}s")
                print(f"  💬 Response: {result['response'][:150]}...")
            else:
                print(f"\n🔹 {algorithm.upper()}: ❌ {result['error']}")
    
    print("\n🎉 Comparison completed!")


if __name__ == "__main__":
    asyncio.run(compare_algorithms())