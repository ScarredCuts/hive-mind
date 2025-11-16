"""Consensus algorithms for synthesizing multiple model responses."""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.response import ConsensusResponse, ModelResponse
from ..reputation.reputation_manager import ReputationManager


class ConsensusAlgorithm(ABC):
    """Abstract base class for consensus algorithms."""
    
    @abstractmethod
    async def synthesize(
        self,
        responses: List[ModelResponse],
        reputation_manager: Optional[ReputationManager] = None
    ) -> ConsensusResponse:
        """Synthesize multiple responses into a consensus."""
        pass


class WeightedVotingConsensus(ConsensusAlgorithm):
    """Weighted voting consensus based on model reputation and confidence."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def synthesize(
        self,
        responses: List[ModelResponse],
        reputation_manager: Optional[ReputationManager] = None
    ) -> ConsensusResponse:
        """Synthesize using weighted voting."""
        if not responses:
            raise ValueError("No responses to synthesize")
        
        if len(responses) == 1:
            return ConsensusResponse(
                content=responses[0].content,
                contributing_responses=[responses[0].model_id],
                confidence_score=responses[0].confidence,
                consensus_strength=1.0,
                divergence_score=0.0,
                synthesis_method="single_response"
            )
        
        # Calculate weights for each response
        weights = self._calculate_weights(responses, reputation_manager)
        
        # Group similar responses
        response_groups = self._group_similar_responses(responses)
        
        # Calculate group weights
        group_weights = self._calculate_group_weights(response_groups, weights)
        
        # Select the best group
        best_group, best_weight = max(group_weights.items(), key=lambda x: x[1])
        
        # Synthesize content from the best group
        synthesized_content = self._synthesize_group_content(best_group)
        
        # Calculate metrics
        confidence_score = self._calculate_confidence(best_group, weights)
        consensus_strength = best_weight / sum(group_weights.values())
        divergence_score = self._calculate_divergence(responses)
        
        return ConsensusResponse(
            content=synthesized_content,
            contributing_responses=[r.model_id for r in best_group],
            confidence_score=confidence_score,
            consensus_strength=consensus_strength,
            divergence_score=divergence_score,
            synthesis_method="weighted_voting"
        )
    
    def _calculate_weights(
        self,
        responses: List[ModelResponse],
        reputation_manager: Optional[ReputationManager]
    ) -> Dict[str, float]:
        """Calculate weights for each response."""
        weights = {}
        
        for response in responses:
            # Base weight from model confidence
            weight = response.confidence
            
            # Apply reputation weight if available
            if reputation_manager:
                reputation_weight = reputation_manager.get_model_weight(response.model_id)
                weight *= reputation_weight
            
            # Apply response time penalty
            if response.response_time_ms > 5000:  # 5 second penalty threshold
                time_penalty = min(0.5, (response.response_time_ms - 5000) / 10000)
                weight *= (1 - time_penalty)
            
            weights[response.model_id] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _group_similar_responses(self, responses: List[ModelResponse]) -> List[List[ModelResponse]]:
        """Group responses by semantic similarity."""
        if len(responses) <= 1:
            return [responses]
        
        # Encode responses
        contents = [r.content for r in responses]
        embeddings = self.encoder.encode(contents)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group using threshold
        groups = []
        used_indices = set()
        
        for i, response in enumerate(responses):
            if i in used_indices:
                continue
            
            # Find similar responses
            group = [response]
            used_indices.add(i)
            
            for j, other_response in enumerate(responses):
                if j in used_indices:
                    continue
                
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    group.append(other_response)
                    used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_group_weights(
        self,
        response_groups: List[List[ModelResponse]],
        weights: Dict[str, float]
    ) -> Dict[List[ModelResponse], float]:
        """Calculate weights for each response group."""
        group_weights = {}
        
        for group in response_groups:
            group_weight = sum(weights[r.model_id] for r in group)
            group_weights[tuple(group)] = group_weight
        
        return group_weights
    
    def _synthesize_group_content(self, group: List[ModelResponse]) -> str:
        """Synthesize content from a group of similar responses."""
        if not group:
            return ""
        
        if len(group) == 1:
            return group[0].content
        
        # Sort by confidence and weight
        sorted_group = sorted(group, key=lambda r: r.confidence, reverse=True)
        
        # Use the highest confidence response as base
        base_content = sorted_group[0].content
        
        # If all responses are very similar, just return the best one
        if len(group) == 2:
            return base_content
        
        # For larger groups, try to merge insights
        return self._merge_similar_contents(sorted_group)
    
    def _merge_similar_contents(self, responses: List[ModelResponse]) -> str:
        """Merge content from similar responses."""
        # Simple strategy: return the highest confidence response
        # In a more sophisticated implementation, we could extract and merge key points
        return responses[0].content
    
    def _calculate_confidence(
        self,
        group: List[ModelResponse],
        weights: Dict[str, float]
    ) -> float:
        """Calculate confidence score for the consensus."""
        if not group:
            return 0.0
        
        # Weighted average of confidences
        total_weight = 0
        weighted_confidence = 0
        
        for response in group:
            weight = weights.get(response.model_id, 0)
            total_weight += weight
            weighted_confidence += response.confidence * weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_divergence(self, responses: List[ModelResponse]) -> float:
        """Calculate how much the responses diverge from each other."""
        if len(responses) <= 1:
            return 0.0
        
        # Encode all responses
        contents = [r.content for r in responses]
        embeddings = self.encoder.encode(contents)
        
        # Calculate average pairwise similarity
        similarity_matrix = cosine_similarity(embeddings)
        n = len(responses)
        
        # Get upper triangle (excluding diagonal)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i][j])
        
        # Divergence is 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1.0 - avg_similarity


class MajorityVotingConsensus(ConsensusAlgorithm):
    """Simple majority voting consensus."""
    
    async def synthesize(
        self,
        responses: List[ModelResponse],
        reputation_manager: Optional[ReputationManager] = None
    ) -> ConsensusResponse:
        """Synthesize using majority voting."""
        if not responses:
            raise ValueError("No responses to synthesize")
        
        if len(responses) == 1:
            return ConsensusResponse(
                content=responses[0].content,
                contributing_responses=[responses[0].model_id],
                confidence_score=responses[0].confidence,
                consensus_strength=1.0,
                divergence_score=0.0,
                synthesis_method="single_response"
            )
        
        # Group similar responses
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        contents = [r.content for r in responses]
        embeddings = encoder.encode(contents)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find the largest group of similar responses
        groups = self._find_majority_groups(responses, similarity_matrix)
        
        if not groups:
            # No clear majority, pick highest confidence
            best_response = max(responses, key=lambda r: r.confidence)
            return ConsensusResponse(
                content=best_response.content,
                contributing_responses=[best_response.model_id],
                confidence_score=best_response.confidence,
                consensus_strength=0.5,
                divergence_score=1.0,
                synthesis_method="highest_confidence"
            )
        
        # Select the largest group
        majority_group = max(groups, key=len)
        
        return ConsensusResponse(
            content=majority_group[0].content,
            contributing_responses=[r.model_id for r in majority_group],
            confidence_score=sum(r.confidence for r in majority_group) / len(majority_group),
            consensus_strength=len(majority_group) / len(responses),
            divergence_score=1.0 - (len(majority_group) / len(responses)),
            synthesis_method="majority_voting"
        )
    
    def _find_majority_groups(
        self,
        responses: List[ModelResponse],
        similarity_matrix: np.ndarray,
        threshold: float = 0.7
    ) -> List[List[ModelResponse]]:
        """Find groups of responses that form a majority."""
        n = len(responses)
        used_indices = set()
        groups = []
        
        for i in range(n):
            if i in used_indices:
                continue
            
            # Find all responses similar to this one
            group = [responses[i]]
            used_indices.add(i)
            
            for j in range(n):
                if j in used_indices:
                    continue
                
                if similarity_matrix[i][j] >= threshold:
                    group.append(responses[j])
                    used_indices.add(j)
            
            if len(group) > 1:  # Only consider groups with multiple responses
                groups.append(group)
        
        return groups


class ConsensusEngine:
    """Main consensus engine that manages different algorithms."""
    
    def __init__(self):
        self.algorithms = {
            "weighted_voting": WeightedVotingConsensus(),
            "majority_voting": MajorityVotingConsensus(),
        }
    
    async def synthesize(
        self,
        responses: List[ModelResponse],
        algorithm: str = "weighted_voting",
        reputation_manager: Optional[ReputationManager] = None
    ) -> ConsensusResponse:
        """Synthesize responses using specified algorithm."""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown consensus algorithm: {algorithm}")
        
        consensus_algorithm = self.algorithms[algorithm]
        return await consensus_algorithm.synthesize(responses, reputation_manager)
    
    def register_algorithm(self, name: str, algorithm: ConsensusAlgorithm) -> None:
        """Register a new consensus algorithm."""
        self.algorithms[name] = algorithm
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available consensus algorithms."""
        return list(self.algorithms.keys())