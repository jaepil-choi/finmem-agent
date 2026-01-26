import pytest
from datetime import datetime
from unittest.mock import MagicMock
from src.rag.strategies.naive import NaiveRAG

class TestNaiveRAGLogic:
    """
    Tests the business logic of NaiveRAG without requiring real FAISS/OpenAI.
    Focuses on date filtering and result merging.
    """
    
    @pytest.fixture
    def rag(self):
        # We mock __init__ to avoid index loading during logic tests
        with MagicMock() as mock_init:
            NaiveRAG.__init__ = lambda self, **kwargs: None
            rag_instance = NaiveRAG()
            return rag_instance

    def test_date_filter_logic(self, rag):
        # Target date: 2024-01-01
        target_date = datetime(2024, 1, 1)
        filter_func = rag._date_filter(target_date)

        # 1. Past date (should pass)
        assert filter_func({"date": "2023-12-31T23:59:59"}) is True
        
        # 2. Same date (should pass)
        assert filter_func({"date": "2024-01-01T00:00:00"}) is True
        
        # 3. Future date (should fail)
        assert filter_func({"date": "2024-01-01T00:00:01"}) is False
        assert filter_func({"date": "2025-01-01T00:00:00"}) is False

        # 4. Missing/Invalid data (should fail gracefully)
        assert filter_func({}) is False
        assert filter_func({"date": "invalid-date"}) is False
