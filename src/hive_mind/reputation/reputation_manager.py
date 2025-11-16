"""Reputation system for model quality tracking."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..models.response import ModelResponse, ResponseQuality


class ReputationEntry:
    """Single reputation entry for a model."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.total_responses = 0
        self.successful_responses = 0
        self.quality_scores: List[float] = []
        self.response_times: List[int] = []
        self.error_count = 0
        self.hallucination_count = 0
        self.last_updated = datetime.utcnow()
        self.weight = 1.0
        
    def update(self, response: ModelResponse) -> None:
        """Update reputation with new response."""
        self.total_responses += 1
        self.response_times.append(response.response_time_ms)
        
        if response.content.startswith("Error:"):
            self.error_count += 1
        elif response.quality_rating == ResponseQuality.HALLUCINATION:
            self.hallucination_count += 1
        else:
            self.successful_responses += 1
        
        if response.quality_score is not None:
            self.quality_scores.append(response.quality_score)
        
        self.last_updated = datetime.utcnow()
        self._recalculate_weight()
    
    def _recalculate_weight(self) -> None:
        """Recalculate model weight based on performance."""
        if self.total_responses == 0:
            self.weight = 1.0
            return
        
        # Base success rate
        success_rate = self.successful_responses / self.total_responses
        
        # Error penalty
        error_penalty = self.error_count / self.total_responses
        
        # Hallucination penalty (more severe)
        hallucination_penalty = (self.hallucination_count / self.total_responses) * 2
        
        # Average response time penalty (responses over 5 seconds)
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        time_penalty = max(0, (avg_response_time - 5000) / 10000)  # Penalty for >5s responses
        
        # Quality score bonus
        quality_bonus = 0
        if self.quality_scores:
            avg_quality = sum(self.quality_scores) / len(self.quality_scores)
            quality_bonus = (avg_quality - 0.5) * 0.5  # Bonus for above-average quality
        
        # Calculate final weight
        self.weight = max(0.1, success_rate - error_penalty - hallucination_penalty - time_penalty + quality_bonus)
        self.weight = min(2.0, self.weight)  # Cap at 2.0
    
    def get_stats(self) -> Dict:
        """Get reputation statistics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        
        return {
            "model_id": self.model_id,
            "total_responses": self.total_responses,
            "success_rate": self.successful_responses / self.total_responses if self.total_responses > 0 else 0,
            "error_rate": self.error_count / self.total_responses if self.total_responses > 0 else 0,
            "hallucination_rate": self.hallucination_count / self.total_responses if self.total_responses > 0 else 0,
            "avg_response_time_ms": avg_response_time,
            "avg_quality_score": avg_quality,
            "weight": self.weight,
            "last_updated": self.last_updated.isoformat()
        }


class ReputationManager:
    """Manages reputation for all models."""
    
    def __init__(self):
        self.reputations: Dict[str, ReputationEntry] = {}
        self.min_responses_for_weight = 5  # Minimum responses before weight is applied
    
    def update_reputation(self, response: ModelResponse) -> None:
        """Update reputation for a model based on response."""
        if response.model_id not in self.reputations:
            self.reputations[response.model_id] = ReputationEntry(response.model_id)
        
        self.reputations[response.model_id].update(response)
    
    def get_model_weight(self, model_id: str) -> float:
        """Get weight for a model based on reputation."""
        if model_id not in self.reputations:
            return 1.0
        
        reputation = self.reputations[model_id]
        
        # Don't apply weight until we have enough data
        if reputation.total_responses < self.min_responses_for_weight:
            return 1.0
        
        return reputation.weight
    
    def get_model_stats(self, model_id: str) -> Optional[Dict]:
        """Get detailed stats for a model."""
        if model_id not in self.reputations:
            return None
        
        return self.reputations[model_id].get_stats()
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all models."""
        return {model_id: rep.get_stats() for model_id, rep in self.reputations.items()}
    
    def set_quality_rating(self, response_id: str, quality: ResponseQuality, score: Optional[float] = None) -> None:
        """Manually set quality rating for a response (for feedback)."""
        # This would need to be implemented with response tracking
        # For now, this is a placeholder for the interface
        pass
    
    def decay_old_reputation(self, days: int = 30) -> None:
        """Apply decay to old reputation data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        for model_id, reputation in self.reputations.items():
            if reputation.last_updated < cutoff_date:
                # Decay the weight towards 1.0
                reputation.weight = reputation.weight * 0.9 + 0.1
    
    def save_to_file(self, filepath: str) -> None:
        """Save reputation data to file."""
        data = {}
        for model_id, reputation in self.reputations.items():
            stats = reputation.get_stats()
            stats["quality_scores"] = reputation.quality_scores
            stats["response_times"] = reputation.response_times
            data[model_id] = stats
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load reputation data from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for model_id, stats in data.items():
                reputation = ReputationEntry(model_id)
                reputation.total_responses = stats["total_responses"]
                reputation.successful_responses = int(stats["total_responses"] * stats["success_rate"])
                reputation.error_count = int(stats["total_responses"] * stats["error_rate"])
                reputation.hallucination_count = int(stats["total_responses"] * stats["hallucination_rate"])
                reputation.quality_scores = stats.get("quality_scores", [])
                reputation.response_times = stats.get("response_times", [])
                reputation.weight = stats["weight"]
                reputation.last_updated = datetime.fromisoformat(stats["last_updated"])
                
                self.reputations[model_id] = reputation
        except FileNotFoundError:
            pass  # No existing reputation data