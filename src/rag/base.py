from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class RAGStrategy(ABC):
    """
    Abstract base class for RAG strategies.
    Ensures that all strategies respect the target_date to prevent look-ahead bias.
    """
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        target_date: datetime, 
        k: int = 5,
        **kwargs: Any
    ) -> List[Document]:
        """
        Retrieve relevant documents for a given query, considering only data 
        available on or before the target_date.
        
        Args:
            query: The search query.
            target_date: The point-in-time for the retrieval. 
                         Only documents with date <= target_date should be retrieved.
            k: Number of documents to retrieve.
            **kwargs: Additional strategy-specific parameters.
            
        Returns:
            List of retrieved LangChain Document objects.
        """
        pass
