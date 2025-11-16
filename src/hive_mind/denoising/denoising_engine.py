"""Denoising engine for iterative response refinement."""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.response import ModelResponse, ResponseQuality
from ..models.model_factory import ModelFactory
from ..models.model_config import ModelConfig


class NoiseDetector:
    """Detects various types of noise in model responses."""
    
    def __init__(self):
        # Patterns that indicate potential issues
        self.hallucination_patterns = [
            r"As an AI language model",
            r"I don't have access to",
            r"I cannot (provide|give|tell)",
            r"I'm not able to",
            r"I don't have personal",
            r"I'm an AI",
            r"As a large language model"
        ]
        
        self.repetition_patterns = [
            r"(.{20,})\1{2,}",  # Repeated sequences
            r"(.{10,})\1{4,}",  # Heavily repeated
        ]
        
        self.contradiction_indicators = [
            "however", "but actually", "on the other hand",
            "contrary to", "in contrast", "although"
        ]
    
    def detect_hallucination(self, content: str) -> float:
        """Detect hallucination indicators in content."""
        score = 0.0
        
        for pattern in self.hallucination_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.3
        
        # Check for very generic or evasive responses
        if len(content) < 20:
            score += 0.2
        
        # Check for excessive apologies
        apology_count = len(re.findall(r"sorry|apologize|regret", content, re.IGNORECASE))
        score += min(0.3, apology_count * 0.1)
        
        return min(1.0, score)
    
    def detect_repetition(self, content: str) -> float:
        """Detect repetitive content."""
        score = 0.0
        
        for pattern in self.repetition_patterns:
            if re.search(pattern, content):
                score += 0.5
        
        # Check word repetition
        words = content.lower().split()
        if words:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # High frequency for common words is normal, check unusual words
            unusual_repetitions = sum(1 for word, count in word_freq.items() 
                                    if count > 3 and len(word) > 4)
            score += min(0.3, unusual_repetitions * 0.1)
        
        return min(1.0, score)
    
    def detect_contradiction(self, content: str) -> float:
        """Detect potential contradictions in content."""
        score = 0.0
        
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) < 2:
            return 0.0
        
        # Look for contradiction indicators
        for indicator in self.contradiction_indicators:
            if indicator in content.lower():
                score += 0.2
        
        # Simple semantic contradiction detection
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            embeddings = encoder.encode([s.strip() for s in sentences if s.strip()])
            if len(embeddings) > 1:
                similarity_matrix = cosine_similarity(embeddings)
                # Low similarity between adjacent sentences might indicate contradiction
                for i in range(len(similarity_matrix) - 1):
                    if similarity_matrix[i][i + 1] < 0.3:
                        score += 0.1
        except:
            pass  # Fallback if encoding fails
        
        return min(1.0, score)
    
    def assess_quality(self, response: ModelResponse) -> Tuple[ResponseQuality, float]:
        """Assess overall quality of a response."""
        hallucination_score = self.detect_hallucination(response.content)
        repetition_score = self.detect_repetition(response.content)
        contradiction_score = self.detect_contradiction(response.content)
        
        # Combined noise score
        noise_score = (hallucination_score + repetition_score + contradiction_score) / 3
        
        # Determine quality rating
        if noise_score > 0.6:
            quality = ResponseQuality.HALLUCINATION
        elif noise_score > 0.4:
            quality = ResponseQuality.POOR
        elif noise_score > 0.2:
            quality = ResponseQuality.FAIR
        elif noise_score > 0.1:
            quality = ResponseQuality.GOOD
        else:
            quality = ResponseQuality.EXCELLENT
        
        # Quality score is inverse of noise
        quality_score = max(0.0, 1.0 - noise_score)
        
        return quality, quality_score


class ResponseRefiner:
    """Refines and improves noisy responses."""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def refine_content(self, content: str, quality: ResponseQuality) -> str:
        """Refine content based on quality assessment."""
        if quality in [ResponseQuality.EXCELLENT, ResponseQuality.GOOD]:
            return content  # No refinement needed
        
        refined = content
        
        # Remove common AI disclaimers
        refined = self._remove_ai_disclaimers(refined)
        
        # Fix repetition
        refined = self._fix_repetition(refined)
        
        # Clean up formatting
        refined = self._clean_formatting(refined)
        
        # If still poor, truncate to most useful part
        if quality == ResponseQuality.HALLUCINATION:
            refined = self._extract_useful_content(refined)
        
        return refined
    
    def _remove_ai_disclaimers(self, content: str) -> str:
        """Remove common AI disclaimers and apologies."""
        patterns = [
            r"As an AI language model[^.]*\.",
            r"I don't have access to[^.]*\.",
            r"I'm sorry,? but I[^.]*\.",
            r"Please note that I[^.]*\.",
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _fix_repetition(self, content: str) -> str:
        """Fix repetitive content."""
        # Remove exact repetitions
        sentences = re.split(r'([.!?]+)', content)
        seen_sentences = set()
        refined_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen_sentences:
                seen_sentences.add(sentence)
                refined_sentences.append(sentence)
            elif not sentence:  # Keep punctuation
                refined_sentences.append(sentence)
        
        return "".join(refined_sentences)
    
    def _clean_formatting(self, content: str) -> str:
        """Clean up formatting issues."""
        # Fix multiple spaces
        content = re.sub(r' +', ' ', content)
        
        # Fix multiple newlines
        content = re.sub(r'\n+', '\n', content)
        
        # Fix punctuation spacing
        content = re.sub(r' ([.!?])', r'\1', content)
        
        return content.strip()
    
    def _extract_useful_content(self, content: str) -> str:
        """Extract the most useful part from a poor response."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Filter out very short or disclaimer-like sentences
        useful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 10 and 
                not any(word in sentence.lower() for word in 
                       ["sorry", "apologize", "cannot", "unable", "ai", "language model"])):
                useful_sentences.append(sentence)
        
        # Return the first useful sentence or a generic fallback
        if useful_sentences:
            return useful_sentences[0] + "."
        
        return "Unable to provide a reliable response."


class DenoisingEngine:
    """Main denoising engine for iterative response refinement."""
    
    def __init__(self):
        self.noise_detector = NoiseDetector()
        self.refiner = ResponseRefiner()
        self.max_iterations = 3
        self.quality_threshold = 0.7
    
    async def denoise_responses(
        self,
        responses: List[ModelResponse],
        max_iterations: Optional[int] = None
    ) -> Tuple[List[ModelResponse], int]:
        """Denoise and refine responses iteratively."""
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        current_responses = responses.copy()
        iterations = 0
        
        for iteration in range(max_iterations):
            iterations += 1
            
            # Assess quality of all responses
            quality_scores = []
            for response in current_responses:
                quality, score = self.noise_detector.assess_quality(response)
                response.quality_rating = quality
                response.quality_score = score
                quality_scores.append(score)
            
            # Check if we've reached acceptable quality
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality >= self.quality_threshold:
                break
            
            # Refine low-quality responses
            refined_responses = []
            for response in current_responses:
                if response.quality_score and response.quality_score < self.quality_threshold:
                    # Refine the content
                    refined_content = self.refiner.refine_content(
                        response.content, 
                        response.quality_rating
                    )
                    
                    # Create new response with refined content
                    refined_response = ModelResponse(
                        model_id=response.model_id,
                        provider=response.provider,
                        content=refined_content,
                        confidence=response.confidence * 0.9,  # Slightly reduce confidence
                        response_time_ms=response.response_time_ms,
                        token_usage=response.token_usage,
                        metadata={**response.metadata, "refined": True, "iteration": iteration}
                    )
                    refined_responses.append(refined_response)
                else:
                    refined_responses.append(response)
            
            current_responses = refined_responses
        
        return current_responses, iterations
    
    def get_denoising_stats(self, original_responses: List[ModelResponse], 
                          denoised_responses: List[ModelResponse]) -> Dict:
        """Get statistics about the denoising process."""
        original_quality = []
        denoised_quality = []
        
        for orig, denoised in zip(original_responses, denoised_responses):
            if orig.quality_score is not None:
                original_quality.append(orig.quality_score)
            if denoised.quality_score is not None:
                denoised_quality.append(denoised.quality_score)
        
        return {
            "original_avg_quality": sum(original_quality) / len(original_quality) if original_quality else 0,
            "denoised_avg_quality": sum(denoised_quality) / len(denoised_quality) if denoised_quality else 0,
            "quality_improvement": (sum(denoised_quality) - sum(original_quality)) / len(original_quality) if original_quality else 0,
            "responses_refined": sum(1 for r in denoised_responses if r.metadata.get("refined", False))
        }