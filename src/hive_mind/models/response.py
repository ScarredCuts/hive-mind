"""Response models for multi-model interactions."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


class ResponseQuality(str, Enum):
    """Quality assessment of model responses."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    HALLUCINATION = "hallucination"


class ModelResponse(BaseModel):
    """Individual model response with metadata."""
    
    model_id: str
    provider: ModelProvider
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    response_time_ms: int
    token_usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_rating: Optional[ResponseQuality] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConsensusResponse(BaseModel):
    """Consensus response synthesized from multiple models."""
    
    content: str
    contributing_responses: List[str]  # Response IDs
    confidence_score: float = Field(ge=0.0, le=1.0)
    consensus_strength: float = Field(ge=0.0, le=1.0)
    divergence_score: float = Field(ge=0.0, le=1.0)
    synthesis_method: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    """Single turn in a conversation."""
    
    turn_id: str
    user_input: str
    model_responses: List[ModelResponse]
    consensus_response: Optional[ConsensusResponse] = None
    denoising_iterations: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context_tokens_used: int = 0


class Conversation(BaseModel):
    """Complete conversation with multiple turns."""
    
    conversation_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new turn to the conversation."""
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()
    
    def get_recent_context(self, max_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns for context."""
        return self.turns[-max_turns:] if self.turns else []