"""Main Hive Mind system that orchestrates multi-model conversations."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..consensus.consensus_engine import ConsensusEngine
from ..denoising.denoising_engine import DenoisingEngine
from ..memory.memory_manager import MemoryManager
from ..models.model_config import HiveMindConfig, ModelConfig
from ..models.model_factory import ModelFactory
from ..models.response import ConsensusResponse, ConversationTurn, ModelResponse
from ..reputation.reputation_manager import ReputationManager
from .conversation import Conversation


class HiveMind:
    """Main Hive Mind system for multi-model conversation orchestration."""
    
    def __init__(self, config: HiveMindConfig):
        self.config = config
        self.conversations: Dict[str, Conversation] = {}
        
        # Initialize components
        self.model_providers = self._initialize_model_providers()
        self.consensus_engine = ConsensusEngine()
        self.denoising_engine = DenoisingEngine()
        self.reputation_manager = ReputationManager()
        self.memory_manager = MemoryManager()
        
        # Load existing memory state
        self.memory_manager.load_memory_state()
    
    def _initialize_model_providers(self) -> Dict[str, any]:
        """Initialize model providers from configuration."""
        providers = {}
        
        for model_config in self.config.get_enabled_models():
            try:
                provider = ModelFactory.create_provider(model_config)
                providers[model_config.model_id] = provider
            except Exception as e:
                print(f"Failed to initialize model {model_config.model_id}: {e}")
        
        return providers
    
    async def process_input(
        self,
        user_input: str,
        conversation_id: Optional[str] = None,
        use_memory: bool = True
    ) -> Dict:
        """Process user input through the Hive Mind system."""
        # Generate or retrieve conversation
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(conversation_id=conversation_id)
            self.conversations[conversation_id] = conversation
        else:
            conversation = self.conversations.get(conversation_id)
            if conversation is None:
                conversation = Conversation(conversation_id=conversation_id)
                self.conversations[conversation_id] = conversation
        
        # Get context if memory is enabled
        context = []
        if use_memory:
            context = self._prepare_context(conversation, user_input)
        
        # Generate responses from all models
        model_responses = await self._generate_model_responses(user_input, context)
        
        # Apply denoising if enabled
        if self.config.denoising_enabled:
            model_responses, denoising_iterations = await self.denoising_engine.denoise_responses(
                model_responses,
                max_iterations=self.config.max_denoising_iterations
            )
        else:
            denoising_iterations = 0
        
        # Generate consensus
        consensus_response = await self.consensus_engine.synthesize(
            model_responses,
            algorithm=self.config.consensus_algorithm,
            reputation_manager=self.reputation_manager if self.config.reputation_enabled else None
        )
        
        # Create conversation turn
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_input=user_input,
            model_responses=model_responses,
            consensus_response=consensus_response,
            denoising_iterations=denoising_iterations,
            context_tokens_used=self._estimate_context_tokens(context)
        )
        
        conversation.add_turn(turn)
        
        # Update reputation system
        if self.config.reputation_enabled:
            for response in model_responses:
                self.reputation_manager.update_reputation(response)
        
        # Store in memory if enabled
        if self.config.memory_enabled:
            self.memory_manager.store_conversation(
                conversation_id,
                user_input,
                consensus_response,
                model_responses,
                {"context_used": len(context) > 0}
            )
        
        return {
            "conversation_id": conversation_id,
            "turn_id": turn.turn_id,
            "response": consensus_response.content,
            "confidence": consensus_response.confidence_score,
            "consensus_strength": consensus_response.consensus_strength,
            "divergence": consensus_response.divergence_score,
            "contributing_models": consensus_response.contributing_responses,
            "denoising_iterations": denoising_iterations,
            "individual_responses": [
                {
                    "model_id": r.model_id,
                    "content": r.content,
                    "confidence": r.confidence,
                    "response_time_ms": r.response_time_ms,
                    "quality_score": r.quality_score
                }
                for r in model_responses
            ]
        }
    
    async def _generate_model_responses(
        self,
        user_input: str,
        context: List[str]
    ) -> List[ModelResponse]:
        """Generate responses from all configured models."""
        tasks = []
        
        for model_id, provider in self.model_providers.items():
            task = self._generate_single_response(provider, user_input, context)
            tasks.append(task)
        
        if self.config.parallel_requests:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            responses = []
            for task in tasks:
                try:
                    response = await task
                    responses.append(response)
                except Exception as e:
                    responses.append(e)
        
        # Filter out exceptions and create error responses
        valid_responses = []
        for i, response in enumerate(responses):
            model_id = list(self.model_providers.keys())[i]
            if isinstance(response, Exception):
                # Create error response
                error_response = ModelResponse(
                    model_id=model_id,
                    provider=self.config.get_model_by_id(model_id).provider,
                    content=f"Error: {str(response)}",
                    confidence=0.0,
                    response_time_ms=0,
                    token_usage={},
                    metadata={"error": str(response)}
                )
                valid_responses.append(error_response)
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    async def _generate_single_response(
        self,
        provider,
        user_input: str,
        context: List[str]
    ) -> ModelResponse:
        """Generate response from a single model provider."""
        try:
            response = await provider.generate_response(user_input, context)
            return response
        except Exception as e:
            # Re-raise to be handled by the calling method
            raise e
    
    def _prepare_context(self, conversation: Conversation, user_input: str) -> List[str]:
        """Prepare context for the models."""
        context_items = []
        
        # Get recent conversation turns
        recent_turns = conversation.get_recent_context(max_turns=3)
        
        for turn in recent_turns:
            context_items.append(f"User: {turn.user_input}")
            if turn.consensus_response:
                context_items.append(f"Assistant: {turn.consensus_response.content}")
        
        # Get similar past conversations from memory
        try:
            memory_context = self.memory_manager.get_context_for_input(user_input, max_context_items=2)
            for ctx in memory_context:
                if ctx["similarity"] > 0.8:  # Only use highly similar contexts
                    context_items.append(f"Previous similar question: {ctx['similar_input']}")
                    context_items.append(f"Previous answer: {ctx['previous_response']}")
        except Exception:
            # If memory retrieval fails, continue without it
            pass
        
        # Check token limits
        total_tokens = sum(self._estimate_tokens(text) for text in context_items)
        max_tokens = self.config.max_context_tokens
        
        if total_tokens > max_tokens:
            # Truncate context to fit within limits
            context_items = self._truncate_context(context_items, max_tokens)
        
        return context_items
    
    def _estimate_context_tokens(self, context: List[str]) -> int:
        """Estimate total tokens in context."""
        return sum(self._estimate_tokens(text) for text in context)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text.split()) * 1.3  # Rough estimate
    
    def _truncate_context(self, context: List[str], max_tokens: int) -> List[str]:
        """Truncate context to fit within token limits."""
        truncated = []
        current_tokens = 0
        
        # Keep most recent items (reverse order)
        for item in reversed(context):
            item_tokens = self._estimate_tokens(item)
            if current_tokens + item_tokens <= max_tokens:
                truncated.insert(0, item)
                current_tokens += item_tokens
            else:
                break
        
        return truncated
    
    async def health_check(self) -> Dict:
        """Check health of all model providers."""
        health_status = {}
        
        for model_id, provider in self.model_providers.items():
            try:
                is_healthy = await provider.health_check()
                health_status[model_id] = {
                    "healthy": is_healthy,
                    "last_check": datetime.utcnow().isoformat()
                }
            except Exception as e:
                health_status[model_id] = {
                    "healthy": False,
                    "error": str(e),
                    "last_check": datetime.utcnow().isoformat()
                }
        
        return health_status
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        reputation_stats = self.reputation_manager.get_all_stats()
        learning_insights = self.memory_manager.get_learning_insights()
        
        return {
            "active_conversations": len(self.conversations),
            "configured_models": len(self.model_providers),
            "reputation_stats": reputation_stats,
            "learning_insights": learning_insights,
            "config": {
                "consensus_algorithm": self.config.consensus_algorithm,
                "denoising_enabled": self.config.denoising_enabled,
                "reputation_enabled": self.config.reputation_enabled,
                "memory_enabled": self.config.memory_enabled
            }
        }
    
    def save_state(self) -> None:
        """Save system state."""
        self.memory_manager.save_memory_state()
        self.reputation_manager.save_to_file("./reputation_data.json")
    
    def load_state(self) -> None:
        """Load system state."""
        self.memory_manager.load_memory_state()
        self.reputation_manager.load_from_file("./reputation_data.json")