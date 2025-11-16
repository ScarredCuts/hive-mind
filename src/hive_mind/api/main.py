"""FastAPI application for Hive Mind system."""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.hive_mind import HiveMind
from ..models.model_config import HiveMindConfig, ModelConfig, ModelProvider


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    conversation_id: Optional[str] = None
    use_memory: bool = True


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    conversation_id: str
    turn_id: str
    response: str
    confidence: float
    consensus_strength: float
    divergence: float
    contributing_models: List[str]
    denoising_iterations: int
    individual_responses: List[Dict]


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    conversation_id: str
    turn_id: str
    satisfaction: float  # 0.0 to 1.0
    feedback: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models: Dict[str, Dict]
    timestamp: datetime


class StatsResponse(BaseModel):
    """Response model for system stats."""
    active_conversations: int
    configured_models: int
    reputation_stats: Dict
    learning_insights: Dict
    config: Dict


# Global Hive Mind instance
hive_mind: Optional[HiveMind] = None

# FastAPI app
app = FastAPI(
    title="Hive Mind API",
    description="Multi-model conversation framework with collective intelligence",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # Connection might be closed, remove it
                self.active_connections.remove(connection)


manager = ConnectionManager()


def initialize_hive_mind(config: HiveMindConfig) -> None:
    """Initialize the global Hive Mind instance."""
    global hive_mind
    hive_mind = HiveMind(config)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hive Mind API - Multi-model conversation framework"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the Hive Mind system."""
    if hive_mind is None:
        raise HTTPException(status_code=500, detail="Hive Mind not initialized")
    
    try:
        result = await hive_mind.process_input(
            user_input=request.message,
            conversation_id=request.conversation_id,
            use_memory=request.use_memory
        )
        
        # Broadcast real-time update
        await manager.broadcast({
            "type": "new_response",
            "conversation_id": result["conversation_id"],
            "turn_id": result["turn_id"],
            "response": result["response"],
            "confidence": result["confidence"]
        })
        
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for learning."""
    if hive_mind is None:
        raise HTTPException(status_code=500, detail="Hive Mind not initialized")
    
    try:
        # Update learning with feedback
        hive_mind.memory_manager.update_with_feedback(
            conversation_id=request.conversation_id,
            user_satisfaction=request.satisfaction,
            feedback=request.feedback
        )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the Hive Mind system."""
    if hive_mind is None:
        raise HTTPException(status_code=500, detail="Hive Mind not initialized")
    
    try:
        health_status = await hive_mind.health_check()
        
        return HealthResponse(
            status="healthy",
            models=health_status,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if hive_mind is None:
        raise HTTPException(status_code=500, detail="Hive Mind not initialized")
    
    try:
        stats = hive_mind.get_system_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    if hive_mind is None:
        raise HTTPException(status_code=500, detail="Hive Mind not initialized")
    
    conversation = hive_mind.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation.conversation_id,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "statistics": conversation.get_statistics(),
        "turns": [
            {
                "turn_id": turn.turn_id,
                "user_input": turn.user_input,
                "response": turn.consensus_response.content if turn.consensus_response else None,
                "confidence": turn.consensus_response.confidence_score if turn.consensus_response else 0,
                "timestamp": turn.timestamp
            }
            for turn in conversation.turns
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time conversation updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or handle WebSocket messages
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/config/models")
async def add_model(model_config: ModelConfig):
    """Add a new model configuration."""
    if hive_mind is None:
        raise HTTPException(status_code=500, detail="Hive Mind not initialized")
    
    try:
        # Add model to configuration
        hive_mind.config.models.append(model_config)
        
        # Initialize new provider
        from ..models.model_factory import ModelFactory
        provider = ModelFactory.create_provider(model_config)
        hive_mind.model_providers[model_config.model_id] = provider
        
        return {"status": "success", "message": f"Model {model_config.model_id} added"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/config/models/{model_id}")
async def remove_model(model_id: str):
    """Remove a model configuration."""
    if hive_mind is None:
        raise HTTPException(status_code=500, detail="Hive Mind not initialized")
    
    try:
        # Remove from configuration
        hive_mind.config.models = [
            m for m in hive_mind.config.models if m.model_id != model_id
        ]
        
        # Remove provider
        if model_id in hive_mind.model_providers:
            del hive_mind.model_providers[model_id]
        
        return {"status": "success", "message": f"Model {model_id} removed"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)