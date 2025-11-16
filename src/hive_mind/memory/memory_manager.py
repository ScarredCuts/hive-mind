"""Memory system for learning from conversation patterns."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.response import ConsensusResponse, ModelResponse
from ..models.model_config import ModelConfig


class ConversationMemory:
    """Stores and retrieves conversation patterns."""
    
    def __init__(self, collection_name: str = "conversations"):
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def store_conversation(
        self,
        conversation_id: str,
        user_input: str,
        consensus_response: ConsensusResponse,
        contributing_responses: List[ModelResponse],
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a conversation turn in memory."""
        # Create embedding for the user input
        embedding = self.encoder.encode(user_input).tolist()
        
        # Prepare metadata
        full_metadata = {
            "conversation_id": conversation_id,
            "user_input": user_input,
            "consensus_response": consensus_response.content,
            "consensus_confidence": consensus_response.confidence_score,
            "consensus_strength": consensus_response.consensus_strength,
            "divergence_score": consensus_response.divergence_score,
            "synthesis_method": consensus_response.synthesis_method,
            "contributing_models": [r.model_id for r in contributing_responses],
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # Store in vector database
        self.collection.add(
            embeddings=[embedding],
            documents=[user_input],
            metadatas=[full_metadata],
            ids=[f"{conversation_id}_{datetime.utcnow().timestamp()}"]
        )
    
    def find_similar_conversations(
        self,
        user_input: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """Find similar past conversations."""
        query_embedding = self.encoder.encode(user_input).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "documents", "distances"]
        )
        
        similar_conversations = []
        
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= min_similarity:
                    similar_conversations.append({
                        "metadata": metadata,
                        "document": results["documents"][0][i],
                        "similarity": similarity
                    })
        
        return similar_conversations
    
    def get_conversation_patterns(
        self,
        time_range_days: int = 30
    ) -> Dict:
        """Analyze conversation patterns over time."""
        # This is a simplified version - in practice you'd want more sophisticated analytics
        cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
        
        # Get all conversations (this would need pagination for large datasets)
        all_results = self.collection.get(include=["metadatas"])
        
        patterns = {
            "total_conversations": 0,
            "avg_consensus_confidence": 0,
            "avg_divergence": 0,
            "most_common_synthesis_method": {},
            "model_usage_frequency": {},
            "high_divergence_conversations": []
        }
        
        if all_results["metadatas"]:
            recent_conversations = []
            confidences = []
            divergences = []
            synthesis_methods = []
            model_usage = {}
            
            for metadata in all_results["metadatas"]:
                conv_date = datetime.fromisoformat(metadata["timestamp"])
                if conv_date >= cutoff_date:
                    recent_conversations.append(metadata)
                    confidences.append(metadata["consensus_confidence"])
                    divergences.append(metadata["divergence_score"])
                    synthesis_methods.append(metadata["synthesis_method"])
                    
                    for model in metadata["contributing_models"]:
                        model_usage[model] = model_usage.get(model, 0) + 1
            
            if recent_conversations:
                patterns["total_conversations"] = len(recent_conversations)
                patterns["avg_consensus_confidence"] = sum(confidences) / len(confidences)
                patterns["avg_divergence"] = sum(divergences) / len(divergences)
                
                # Most common synthesis method
                method_counts = {}
                for method in synthesis_methods:
                    method_counts[method] = method_counts.get(method, 0) + 1
                patterns["most_common_synthesis_method"] = method_counts
                
                patterns["model_usage_frequency"] = model_usage
                
                # High divergence conversations (potential learning opportunities)
                high_divergence = [
                    conv for conv in recent_conversations 
                    if conv["divergence_score"] > 0.7
                ]
                patterns["high_divergence_conversations"] = high_divergence[:5]  # Top 5
        
        return patterns


class LearningEngine:
    """Learns from conversation patterns to improve consensus algorithms."""
    
    def __init__(self):
        self.learning_data = {
            "successful_patterns": [],
            "failed_patterns": [],
            "model_performance": {},
            "synthesis_method_effectiveness": {}
        }
        self.min_samples_for_learning = 10
    
    def record_outcome(
        self,
        user_input: str,
        consensus_response: ConsensusResponse,
        user_satisfaction: Optional[float] = None,
        feedback: Optional[str] = None
    ) -> None:
        """Record the outcome of a conversation for learning."""
        outcome = {
            "user_input": user_input,
            "consensus_response": consensus_response.content,
            "synthesis_method": consensus_response.synthesis_method,
            "consensus_strength": consensus_response.consensus_strength,
            "divergence_score": consensus_response.divergence_score,
            "confidence_score": consensus_response.confidence_score,
            "contributing_models": consensus_response.contributing_responses,
            "timestamp": datetime.utcnow().isoformat(),
            "user_satisfaction": user_satisfaction,
            "feedback": feedback
        }
        
        # Classify as successful or failed based on available metrics
        if user_satisfaction is not None:
            if user_satisfaction >= 0.7:
                self.learning_data["successful_patterns"].append(outcome)
            elif user_satisfaction < 0.4:
                self.learning_data["failed_patterns"].append(outcome)
        else:
            # Use proxy metrics if no explicit feedback
            if (consensus_response.consensus_strength > 0.8 and 
                consensus_response.confidence_score > 0.7 and
                consensus_response.divergence_score < 0.3):
                self.learning_data["successful_patterns"].append(outcome)
            elif consensus_response.divergence_score > 0.7:
                self.learning_data["failed_patterns"].append(outcome)
    
    def get_synthesis_recommendations(self) -> Dict[str, float]:
        """Get recommendations for synthesis method weights."""
        recommendations = {}
        
        # Analyze effectiveness of different synthesis methods
        method_stats = {}
        
        for pattern in self.learning_data["successful_patterns"]:
            method = pattern["synthesis_method"]
            if method not in method_stats:
                method_stats[method] = {"success": 0, "total": 0}
            method_stats[method]["success"] += 1
            method_stats[method]["total"] += 1
        
        for pattern in self.learning_data["failed_patterns"]:
            method = pattern["synthesis_method"]
            if method not in method_stats:
                method_stats[method] = {"success": 0, "total": 0}
            method_stats[method]["total"] += 1
        
        # Calculate success rates
        for method, stats in method_stats.items():
            if stats["total"] >= self.min_samples_for_learning:
                success_rate = stats["success"] / stats["total"]
                recommendations[method] = success_rate
        
        return recommendations
    
    def get_model_insights(self) -> Dict:
        """Get insights about model performance patterns."""
        insights = {
            "high_performing_models": [],
            "problematic_combinations": [],
            "optimal_group_sizes": {}
        }
        
        # Analyze successful patterns for model combinations
        successful_combinations = {}
        for pattern in self.learning_data["successful_patterns"]:
            models = tuple(sorted(pattern["contributing_models"]))
            successful_combinations[models] = successful_combinations.get(models, 0) + 1
        
        # Find most successful combinations
        if successful_combinations:
            sorted_combinations = sorted(successful_combinations.items(), key=lambda x: x[1], reverse=True)
            insights["high_performing_models"] = [
                {"models": list(combo), "success_count": count}
                for combo, count in sorted_combinations[:5]
            ]
        
        return insights
    
    def save_learning_data(self, filepath: str) -> None:
        """Save learning data to file."""
        with open(filepath, 'w') as f:
            json.dump(self.learning_data, f, indent=2)
    
    def load_learning_data(self, filepath: str) -> None:
        """Load learning data from file."""
        try:
            with open(filepath, 'r') as f:
                self.learning_data = json.load(f)
        except FileNotFoundError:
            pass  # No existing learning data


class MemoryManager:
    """Main memory manager that coordinates all memory components."""
    
    def __init__(self, vector_db_path: str = "./chroma_db"):
        self.conversation_memory = ConversationMemory()
        self.learning_engine = LearningEngine()
        self.vector_db_path = vector_db_path
    
    def store_conversation(
        self,
        conversation_id: str,
        user_input: str,
        consensus_response: ConsensusResponse,
        contributing_responses: List[ModelResponse],
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a conversation and update learning."""
        # Store in conversation memory
        self.conversation_memory.store_conversation(
            conversation_id, user_input, consensus_response, contributing_responses, metadata
        )
        
        # Record for learning (without user satisfaction for now)
        self.learning_engine.record_outcome(user_input, consensus_response)
    
    def get_context_for_input(
        self,
        user_input: str,
        max_context_items: int = 3
    ) -> List[Dict]:
        """Get relevant context for a new user input."""
        similar_conversations = self.conversation_memory.find_similar_conversations(
            user_input, limit=max_context_items
        )
        
        context = []
        for conv in similar_conversations:
            context.append({
                "similar_input": conv["document"],
                "previous_response": conv["metadata"]["consensus_response"],
                "similarity": conv["similarity"],
                "confidence": conv["metadata"]["consensus_confidence"]
            })
        
        return context
    
    def get_learning_insights(self) -> Dict:
        """Get insights from the learning engine."""
        synthesis_recs = self.learning_engine.get_synthesis_recommendations()
        model_insights = self.learning_engine.get_model_insights()
        patterns = self.conversation_memory.get_conversation_patterns()
        
        return {
            "synthesis_method_recommendations": synthesis_recs,
            "model_performance_insights": model_insights,
            "conversation_patterns": patterns
        }
    
    def update_with_feedback(
        self,
        conversation_id: str,
        user_satisfaction: float,
        feedback: Optional[str] = None
    ) -> None:
        """Update learning with user feedback."""
        # This would need to be implemented with proper conversation tracking
        # For now, this is a placeholder interface
        pass
    
    def save_memory_state(self, base_path: str = "./memory") -> None:
        """Save all memory state to files."""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        self.learning_engine.save_learning_data(f"{base_path}/learning_data.json")
    
    def load_memory_state(self, base_path: str = "./memory") -> None:
        """Load memory state from files."""
        import os
        
        if os.path.exists(f"{base_path}/learning_data.json"):
            self.learning_engine.load_learning_data(f"{base_path}/learning_data.json")