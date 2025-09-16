"""
Response Evaluation Module
Provides comprehensive evaluation metrics for AI responses
"""

import re
from typing import Dict, List, Optional
from datetime import datetime


class ResponseEvaluator:
    """Evaluate AI responses with multiple metrics"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_response(self, 
                         query: str, 
                         response: str, 
                         context: str = "", 
                         pdf_chunks_info: List[Dict] = None) -> Dict:
        """Comprehensive response evaluation"""
        
        evaluation = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "query": query,
            "response": response,
            "context_used": bool(context),
            "metrics": {}
        }
        
        # Basic response metrics
        evaluation["metrics"]["response_length"] = len(response)
        evaluation["metrics"]["word_count"] = len(response.split())
        evaluation["metrics"]["sentence_count"] = len(re.split(r'[.!?]+', response))
        
        # Content quality metrics
        evaluation["metrics"]["readability_score"] = self._calculate_readability(response)
        evaluation["metrics"]["coherence_score"] = self._calculate_coherence(response)
        evaluation["metrics"]["completeness_score"] = self._calculate_completeness(query, response)
        
        # Context utilization metrics (if PDF context available)
        if context and pdf_chunks_info:
            context_metrics = self._evaluate_context_usage(query, response, context, pdf_chunks_info)
            evaluation["metrics"].update(context_metrics)
        
        # Query relevance
        evaluation["metrics"]["relevance_score"] = self._calculate_relevance(query, response)
        
        # Overall score
        evaluation["metrics"]["overall_score"] = self._calculate_overall_score(evaluation["metrics"])
        
        # Add evaluation to history
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score based on sentence and word length"""
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (lower is better for readability)
        # Normalize to 0-10 scale (10 being most readable)
        complexity = (avg_sentence_length * 0.39) + (avg_word_length * 11.8)
        readability = max(0, min(10, 15 - (complexity / 10)))
        
        return round(readability, 2)
    
    def _calculate_coherence(self, text: str) -> float:
        """Measure response coherence based on structure and flow"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 8.0  # Short responses are generally coherent
        
        # Check for transition words and logical connectors
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 
                          'consequently', 'nevertheless', 'meanwhile', 'specifically', 'for example']
        
        transition_count = sum(1 for word in transition_words if word in text.lower())
        transition_ratio = transition_count / len(sentences)
        
        # Check for repetitive sentence structures
        sentence_starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        unique_starts = len(set(sentence_starts))
        variety_ratio = unique_starts / len(sentences) if sentences else 0
        
        # Combine metrics (0-10 scale)
        coherence = (transition_ratio * 3 + variety_ratio * 7) * 10
        return round(min(10, max(0, coherence)), 2)
    
    def _calculate_completeness(self, query: str, response: str) -> float:
        """Evaluate how completely the response addresses the query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words for better analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_content_words = query_words - stop_words
        
        if not query_content_words:
            return 8.0
        
        # Check how many query concepts are addressed
        addressed_concepts = len(query_content_words.intersection(response_words))
        completeness = (addressed_concepts / len(query_content_words)) * 10
        
        return round(min(10, max(0, completeness)), 2)
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate how relevant the response is to the query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        query_content = query_words - stop_words
        response_content = response_words - stop_words
        
        if not query_content:
            return 7.0
        
        # Calculate overlap and semantic similarity
        overlap = len(query_content.intersection(response_content))
        overlap_ratio = overlap / len(query_content)
        
        # Boost score if response is focused (not too wordy for simple queries)
        length_penalty = max(0, (len(response.split()) - 50) / 100) if len(query.split()) < 10 else 0
        
        relevance = (overlap_ratio * 10) - length_penalty
        return round(min(10, max(0, relevance)), 2)
    
    def _evaluate_context_usage(self, query: str, response: str, context: str, chunks_info: List[Dict]) -> Dict:
        """Evaluate how well the context (PDF) was utilized"""
        context_metrics = {}
        
        if not chunks_info:
            return context_metrics
        
        # Context utilization score
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        context_overlap = len(context_words.intersection(response_words))
        
        # Evaluate chunk relevance scores
        avg_chunk_relevance = sum(chunk["relevance_score"] for chunk in chunks_info) / len(chunks_info)
        max_chunk_relevance = max(chunk["relevance_score"] for chunk in chunks_info)
        
        context_metrics.update({
            "context_utilization": round(min(10, (context_overlap / len(context_words)) * 100), 2),
            "avg_chunk_relevance": round(avg_chunk_relevance, 2),
            "max_chunk_relevance": round(max_chunk_relevance, 2),
            "chunks_used": len(chunks_info),
            "total_matching_words": sum(chunk["keyword_matches"] for chunk in chunks_info)
        })
        
        return context_metrics
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate overall quality score"""
        # Weight different metrics
        weights = {
            "relevance_score": 0.3,
            "completeness_score": 0.25,
            "coherence_score": 0.2,
            "readability_score": 0.15,
            "context_utilization": 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        # If context wasn't used, redistribute its weight
        if "context_utilization" not in metrics and total_weight < 1.0:
            remaining_weight = 1.0 - total_weight
            total_score += 7.0 * remaining_weight  # Neutral score for missing context
        
        return round(min(10, max(0, total_score)), 2)
    
    def get_score_explanation(self) -> Dict[str, str]:
        """Get explanations for all scoring metrics"""
        return {
            "overall_score": "Weighted average of all metrics (0-10, 10 being best)",
            "relevance_score": "How well the response addresses the query (0-10)",
            "completeness_score": "How thoroughly the query concepts are covered (0-10)",
            "coherence_score": "Logical flow and structure of the response (0-10)",
            "readability_score": "Ease of reading and understanding (0-10)",
            "context_utilization": "How effectively PDF context was used (0-10)",
            "avg_chunk_relevance": "Average relevance of selected PDF chunks",
            "max_chunk_relevance": "Highest relevance score among selected chunks",
            "chunks_used": "Number of PDF chunks used for context",
            "word_count": "Total words in the response",
            "sentence_count": "Total sentences in the response"
        }
