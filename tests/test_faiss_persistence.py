import os
import shutil
import pytest
from datetime import datetime
from pathlib import Path
from langchain_core.documents import Document
from src.rag.strategies.finmem import FinMemRAG
from src.db.loaders.faiss_loader import FAISSLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.config import settings

from langchain_core.embeddings import Embeddings

class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing without network access."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 1536

@pytest.fixture
def temp_vector_db():
    # Setup a temporary vector DB path
    temp_dir = Path("data/test_vector_db")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    
    # Use FakeEmbeddings instead of OpenAI
    embeddings = FakeEmbeddings()
    
    doc = Document(
        page_content="Apple reported strong earnings in Q4 2023.",
        metadata={
            "source_id": "test_123",
            "date": "2023-12-01T00:00:00",
            "importance": 40.0,
            "access_counter": 0,
            "tier": "shallow"
        }
    )
    
    tier_path = temp_dir / "shallow"
    tier_path.mkdir()
    
    vectorstore = FAISS.from_documents([doc], embeddings)
    vectorstore.save_local(str(tier_path))
    
    yield temp_dir
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def test_faiss_metadata_persistence(temp_vector_db):
    """
    Test if updating metadata in docstore and calling save_local() 
    actually persists the changes.
    """
    # 1. Initialize strategy with temp DB
    strategy = FinMemRAG(vector_db_root=str(temp_vector_db))
    tier = "shallow"
    index = strategy.indices[tier]
    
    # 2. Update metadata manually (simulating reflection_node logic)
    target_source_id = "test_123"
    target_doc = None
    for internal_id, doc in index.docstore._dict.items():
        if doc.metadata.get("source_id") == target_source_id:
            target_doc = doc
            doc.metadata["importance"] = 99.0
            doc.metadata["access_counter"] = 5
            break
    
    assert target_doc is not None, "Target document not found in docstore"
    
    # 3. Save local
    tier_path = temp_vector_db / tier
    index.save_local(str(tier_path))
    
    # 4. Reload from disk
    embeddings = FakeEmbeddings()
    new_index = FAISS.load_local(
        str(tier_path), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 5. Verify persistence
    found = False
    for internal_id, doc in new_index.docstore._dict.items():
        if doc.metadata.get("source_id") == target_source_id:
            assert doc.metadata["importance"] == 99.0
            assert doc.metadata["access_counter"] == 5
            found = True
            break
            
    assert found, "Document not found after reload"
    print("\n✅ FAISS Metadata Persistence Verified.")
