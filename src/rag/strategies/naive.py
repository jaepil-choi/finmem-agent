import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.rag.base import RAGStrategy

class NaiveRAG(RAGStrategy):
    """
    Standard similarity-based RAG strategy with strict date filtering.
    Retrieves from all available tiers (shallow, intermediate, archive).
    """
    
    def __init__(self, vector_db_root: Optional[str] = None):
        self.vector_db_root = Path(vector_db_root or settings.VECTOR_DB_ROOT)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )
        self.tiers = ["shallow", "intermediate", "archive"]
        self.indices = self._load_indices()
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def _load_indices(self) -> Dict[str, FAISS]:
        """Loads FAISS indices for all tiers."""
        indices = {}
        for tier in self.tiers:
            tier_path = self.vector_db_root / tier
            if (tier_path / "index.faiss").exists():
                try:
                    indices[tier] = FAISS.load_local(
                        str(tier_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logging.info(f"Loaded FAISS index for tier: {tier}")
                except Exception as e:
                    logging.error(f"Failed to load FAISS index for {tier}: {e}")
            else:
                logging.warning(f"No FAISS index found for tier: {tier}")
        return indices

    def _date_filter(self, target_date: datetime):
        """
        Creates a filter function for FAISS metadata.
        Ensures document date <= target_date.
        """
        def filter_func(metadata: Dict[str, Any]) -> bool:
            doc_date_str = metadata.get("date")
            if not doc_date_str:
                return False
            try:
                # Metadata dates are stored as ISO format strings in FAISSLoader
                doc_date = datetime.fromisoformat(doc_date_str)
                return doc_date <= target_date
            except (ValueError, TypeError):
                return False
        return filter_func

    def retrieve(
        self, 
        query: str, 
        target_date: datetime, 
        k: int = 5,
        **kwargs: Any
    ) -> List[Document]:
        """
        Retrieves top k documents across all tiers that match the date constraint.
        """
        all_results = []
        filter_func = self._date_filter(target_date)
        
        # 1. Retrieve candidates from each tier
        # We retrieve k from each tier to ensure we have enough candidates after merging
        for tier, index in self.indices.items():
            try:
                # FAISS similarity_search_with_score returns (Document, score)
                # Lower score is better for FAISS (L2 distance)
                results_with_score = index.similarity_search_with_score(
                    query, 
                    k=k, 
                    filter=filter_func
                )
                for doc, score in results_with_score:
                    # Add tier info to metadata if not present
                    doc.metadata["tier"] = tier
                    doc.metadata["score"] = float(score)
                    all_results.append(doc)
            except Exception as e:
                self.logger.error(f"Error retrieving from tier {tier}: {e}")

        # 2. Sort all candidates by score (similarity)
        # Note: FAISS scores are distances, so lower is more similar
        all_results.sort(key=lambda x: x.metadata["score"])

        # 3. Return top k overall
        return all_results[:k]
