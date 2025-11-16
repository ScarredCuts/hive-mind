"""Conversation management for Hive Mind."""

from datetime import datetime
from typing import Dict, List, Optional

from ..models.response import ConversationTurn, ModelResponse, ConsensusResponse


class Conversation:
    """Manages a single conversation session."""
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.turns: List[ConversationTurn] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.metadata: Dict = {}
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new turn to the conversation."""
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()
    
    def get_recent_turns(self, max_turns: int = 5) -> List[ConversationTurn]:
        """Get the most recent conversation turns."""
        return self.turns[-max_turns:] if self.turns else []
    
    def get_turn_by_id(self, turn_id: str) -> Optional[ConversationTurn]:
        """Get a specific turn by ID."""
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None
    
    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        if not self.turns:
            return ""
        
        summary_parts = []
        for turn in self.turns[-3:]:  # Last 3 turns
            summary_parts.append(f"User: {turn.user_input}")
            if turn.consensus_response:
                summary_parts.append(f"Assistant: {turn.consensus_response.content}")
        
        return "\n".join(summary_parts)
    
    def get_statistics(self) -> Dict:
        """Get conversation statistics."""
        if not self.turns:
            return {
                "total_turns": 0,
                "avg_confidence": 0,
                "avg_consensus_strength": 0,
                "models_used": [],
                "total_denoising_iterations": 0
            }
        
        confidences = []
        consensus_strengths = []
        models_used = set()
        total_denoising_iterations = 0
        
        for turn in self.turns:
            if turn.consensus_response:
                confidences.append(turn.consensus_response.confidence_score)
                consensus_strengths.append(turn.consensus_response.consensus_strength)
                models_used.update(turn.consensus_response.contributing_responses)
            
            total_denoising_iterations += turn.denoising_iterations
        
        return {
            "total_turns": len(self.turns),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "avg_consensus_strength": sum(consensus_strengths) / len(consensus_strengths) if consensus_strengths else 0,
            "models_used": list(models_used),
            "total_denoising_iterations": total_denoising_iterations,
            "duration_minutes": (self.updated_at - self.created_at).total_seconds() / 60
        }