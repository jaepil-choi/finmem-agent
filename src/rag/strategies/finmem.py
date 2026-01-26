import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from src.rag.strategies.naive import NaiveRAG

class FinMemRAG(NaiveRAG):
    """
    FinMem-style RAG strategy that combines Semantic Similarity, 
    Recency, and Importance for document re-ranking.
    """
    
    def __init__(self, vector_db_root: Optional[str] = None, tau: float = 7.0):
        super().__init__(vector_db_root)
        self.tau = tau # Decay constant for recency (days)

    def _calculate_recency_score(self, doc_date: datetime, target_date: datetime) -> float:
        """
        Calculates exponential decay recency score: e^(-delta_t / tau)
        """
        delta_t = (target_date - doc_date).days
        if delta_t < 0:
            return 0.0 # Future document (should be filtered out anyway)
        return np.exp(-delta_t / self.tau)

    def _calculate_importance_score(self, importance: float) -> float:
        """
        Normalizes importance score: min(importance, 100) / 100
        """
        return min(max(importance, 0), 100) / 100.0

    def retrieve(
        self, 
        query: str, 
        target_date: datetime, 
        k: int = 5,
        **kwargs: Any
    ) -> List[Document]:
        """
        Retrieves documents using similarity search and re-ranks them 
        using the FinMem composite scoring logic.
        """
        # 1. Retrieve more candidates than k to allow for re-ranking
        candidate_k = k * 3 
        all_candidates = super().retrieve(query, target_date, k=candidate_k)
        
        if not all_candidates:
            return []

        # 2. Re-rank candidates
        scored_candidates = []
        for doc in all_candidates:
            # S: Similarity Score
            # FAISS score is distance (lower is better). 
            # We convert it to a similarity proxy [0, 1] if it's L2 distance.
            # If it's already a similarity score, we use it directly.
            # For text-embedding-3-small, scores are often < 1.0.
            # We use 1 / (1 + distance) as a stable similarity mapping for L2.
            dist = doc.metadata.get("score", 1.0)
            similarity_s = 1.0 / (1.0 + dist)
            
            # R: Recency Score
            doc_date_str = doc.metadata.get("date")
            try:
                doc_date = datetime.fromisoformat(doc_date_str)
                recency_r = self._calculate_recency_score(doc_date, target_date)
            except (ValueError, TypeError):
                recency_r = 0.0
                
            # I: Importance Score
            importance_val = doc.metadata.get("importance", 40.0)
            importance_i = self._calculate_importance_score(importance_val)
            
            # Composite Score: S + R + I
            composite_score = similarity_s + recency_r + importance_i
            
            doc.metadata["finmem_scores"] = {
                "similarity": float(similarity_s),
                "recency": float(recency_r),
                "importance": float(importance_i),
                "composite": float(composite_score)
            }
            scored_candidates.append((doc, composite_score))

        # 3. Sort by composite score (higher is better)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 4. Return top k
        return [doc for doc, score in scored_candidates[:k]]
