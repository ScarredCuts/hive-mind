# 🧠 Hive Mind

Multi-model conversation framework with collective intelligence that uses ensemble AI techniques to denoise and refine outputs through consensus algorithms.

## 🌟 Features

### 🤖 Multi-Model Architecture
- **Parallel Processing**: Orchestrate multiple AI models (OpenAI, Anthropic, Cohere) simultaneously
- **Provider Agnostic**: Support for all major LLM APIs with extensible architecture
- **Dynamic Model Management**: Add/remove models at runtime without system restart

### 🔄 Denoising Engine
- **Iterative Refinement**: Multiple rounds of cross-validation between models
- **Noise Detection**: Identifies hallucinations, repetitions, and contradictions
- **Quality Assessment**: Automatic quality scoring and content refinement
- **Adaptive Filtering**: Learns to filter out low-quality responses over time

### 🛡️ Bad Actor Detection & Reputation System
- **Performance Tracking**: Monitors response quality, speed, and reliability
- **Dynamic Weighting**: Automatically adjusts model influence based on performance
- **Error Detection**: Identifies models prone to hallucinations or failures
- **Reputation Decay**: Older performance data gradually loses influence

### 🧠 Evolutionary Learning
- **Pattern Recognition**: Learns from successful conversation patterns
- **Memory Storage**: Vector database for efficient similarity search
- **Continuous Improvement**: Gets smarter with each interaction
- **Contextual Learning**: Remembers what works in specific domains

### 🤝 Consensus Algorithms
- **Weighted Voting**: Reputation-weighted decision making
- **Majority Voting**: Simple democratic consensus
- **Similarity Grouping**: Groups semantically similar responses
- **Custom Algorithms**: Extensible framework for new consensus methods

### 📝 Context Management
- **Token Optimization**: Efficient context window management
- **Memory Integration**: Retrieves relevant past conversations
- **Conversation History**: Maintains coherent multi-turn dialogues
- **Context Truncation**: Smart context prioritization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hive-mind

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export COHERE_API_KEY="your-cohere-key"
```

### Basic Usage

```python
import asyncio
from src.hive_mind.core.hive_mind import HiveMind
from src.hive_mind.utils.config_loader import load_config_from_file

async def main():
    # Load configuration
    config = load_config_from_file("config/default_config.yaml")
    
    # Initialize Hive Mind
    hive_mind = HiveMind(config)
    
    # Process a query
    result = await hive_mind.process_input(
        user_input="What is quantum computing?",
        use_memory=True
    )
    
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Consensus Strength: {result['consensus_strength']}")

asyncio.run(main())
```

### Web Interface

Start the web server:

```bash
python main.py
```

Open `http://localhost:8000/web/index.html` in your browser for the real-time conversation interface.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   Model Layer    │───▶│  Raw Responses  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Context       │◀───│   Memory Layer   │───▶│  Denoising      │
│   Management    │    └──────────────────┘    │  Engine         │
└─────────────────┘                            └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Reputation    │◀───│  Consensus       │◀───│  Consensus      │
│   System        │    │  Algorithms      │    │  Response       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Core Components

### Model Providers
- **OpenAI**: GPT-3.5, GPT-4, and future models
- **Anthropic**: Claude series models
- **Cohere**: Command and generation models
- **Extensible**: Easy to add new providers

### Consensus Algorithms
- **Weighted Voting**: Models weighted by reputation and confidence
- **Majority Voting**: Simple majority rule with similarity grouping
- **Hybrid Methods**: Combine multiple approaches
- **Custom Algorithms**: Implement your own consensus logic

### Denoising Pipeline
1. **Quality Assessment**: Score each response for quality issues
2. **Noise Detection**: Identify hallucinations, repetitions, contradictions
3. **Content Refinement**: Clean and improve low-quality responses
4. **Iterative Improvement**: Multiple rounds until quality threshold met

### Reputation System
- **Performance Metrics**: Response quality, speed, error rate
- **Dynamic Weights**: Automatically adjusts model influence
- **Learning Component**: Improves accuracy over time
- **Bad Actor Detection**: Identifies and downweights unreliable models

## 🔧 Configuration

### Model Configuration

```yaml
models:
  - model_id: "gpt-4"
    provider: "openai"
    model_name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    max_tokens: 2048
    temperature: 0.7
    enabled: true
    weight: 1.0
```

### System Configuration

```yaml
# Consensus settings
consensus_algorithm: "weighted_voting"
consensus_threshold: 0.7

# Denoising settings
denoising_enabled: true
max_denoising_iterations: 3

# Reputation system
reputation_enabled: true

# Memory system
memory_enabled: true
vector_db_path: "./chroma_db"
```

## 🌐 API Endpoints

### Chat
```http
POST /chat
{
  "message": "What is AI?",
  "conversation_id": "optional-conversation-id",
  "use_memory": true
}
```

### Health Check
```http
GET /health
```

### System Statistics
```http
GET /stats
```

### Feedback
```http
POST /feedback
{
  "conversation_id": "conv-id",
  "turn_id": "turn-id", 
  "satisfaction": 0.8,
  "feedback": "Great response!"
}
```

## 📈 Monitoring & Analytics

### System Metrics
- Active conversations
- Model health status
- Response times
- Consensus quality scores

### Learning Insights
- Most effective model combinations
- Consensus algorithm performance
- Conversation pattern analysis
- Quality improvement trends

### Reputation Analytics
- Model performance rankings
- Error rate analysis
- Response time distributions
- Quality score trends

## 🧪 Examples

### Basic Usage
```bash
python examples/basic_usage.py
```

### Algorithm Comparison
```bash
python examples/compare_algorithms.py
```

### Performance Analysis
```bash
python examples/performance_analysis.py
```

## 🔬 Advanced Features

### Custom Consensus Algorithms
```python
from src.hive_mind.consensus.consensus_engine import ConsensusAlgorithm

class CustomConsensus(ConsensusAlgorithm):
    async def synthesize(self, responses, reputation_manager=None):
        # Your custom logic here
        pass

# Register your algorithm
hive_mind.consensus_engine.register_algorithm("custom", CustomConsensus())
```

### Custom Model Providers
```python
from src.hive_mind.models.base_model import BaseModelProvider

class CustomProvider(BaseModelProvider):
    async def generate_response(self, prompt, context=None):
        # Your model integration here
        pass

# Register your provider
ModelFactory.register_provider("custom", CustomProvider)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 Requirements

- Python 3.9+
- Redis (for caching)
- ChromaDB (for vector storage)
- API keys for supported LLM providers

## 🗄️ Dependencies

See `requirements.txt` for full list of dependencies.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models  
- Cohere for Command models
- Sentence Transformers for embeddings
- ChromaDB for vector storage
- FastAPI for the web framework

## 🔮 Future Roadmap

- [ ] Support for more model providers (Hugging Face, local models)
- [ ] Advanced consensus algorithms (neural voting, ensemble methods)
- [ ] Real-time conversation streaming
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Model fine-tuning integration
- [ ] Distributed processing capabilities
- [ ] GraphQL API
- [ ] Mobile app interface

---

Built with ❤️ by the Hive Mind team