import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.db.mongodb_client import MongoDBClient

class FAISSLoader:
    def __init__(self, db_name=None, vector_db_root=None):
        self.db_name = db_name or settings.MONGODB_DB_NAME
        self.vector_db_root = Path(vector_db_root or settings.VECTOR_DB_ROOT)
        self.vector_db_root.mkdir(parents=True, exist_ok=True)
        
        self.mongo = MongoDBClient(db_name=self.db_name)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
            chunk_size=100
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            add_start_index=True
        )
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)

    def get_tier_for_category(self, category: str) -> str:
        if category == "daily":
            return "shallow"
        if category in ["weekly", "monthly"]:
            return "intermediate"
        raise ValueError(f"Unknown category: {category}. Expected 'daily', 'weekly', or 'monthly'.")

    def process_collection(self, collection_name: str, limit: Optional[int] = None, dry_run: bool = False, batch_size: int = 50):
        tier = self.get_tier_for_category(collection_name)
        self.logger.info(f"Syncing MongoDB collection '{collection_name}' to FAISS tier '{tier}'")
        
        # Paper-based initial Importance assignment
        # Initializing with 40 as a baseline, and 0 for access counter.
        initial_importance = 40.0
        initial_access_counter = 0

        # 1. Fetch documents from MongoDB
        query = self.mongo.db[collection_name].find()
        if limit:
            query = query.limit(limit)
            
        mongo_docs = list(query)
        if not mongo_docs:
            self.logger.info(f"No documents found in collection '{collection_name}'")
            return

        # 2 & 3. Convert and Chunking
        all_chunks = []
        for m_doc in tqdm(mongo_docs, desc=f"Chunking {collection_name}"):
            doc = Document(
                page_content=m_doc["full_text"],
                metadata={
                    "source_id": str(m_doc["_id"]),
                    "date": m_doc["date"].isoformat() if isinstance(m_doc["date"], datetime) else m_doc["date"],
                    "filename": m_doc.get("filename", "unknown"),
                    "category": collection_name,
                    "tier": tier,
                    "importance": initial_importance,
                    "access_counter": initial_access_counter,
                    **m_doc.get("metadata", {})
                }
            )
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        if dry_run:
            self.logger.info(f"[Dry-run] Would have saved {len(all_chunks)} chunks to tier '{tier}'")
            return

        # 4. Create or Update FAISS index in Batches
        tier_path = self.vector_db_root / tier
        vectorstore = None
        
        if (tier_path / "index.faiss").exists():
            vectorstore = FAISS.load_local(
                str(tier_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        
        self.logger.info(f"Embedding and adding {len(all_chunks)} chunks to FAISS...")
        for i in tqdm(range(0, len(all_chunks), batch_size), desc=f"Embedding {tier}"):
            batch = all_chunks[i : i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                vectorstore.add_documents(batch)
        
        # 5. Save locally
        if vectorstore:
            vectorstore.save_local(str(tier_path))
            self.logger.info(f"Successfully saved FAISS index for tier '{tier}'")

    def sync_all(self, limit: Optional[int] = None, dry_run: bool = False, batch_size: int = 50):
        for collection in ["daily", "weekly", "monthly"]:
            self.process_collection(collection, limit=limit, dry_run=dry_run, batch_size=batch_size)

    def close(self):
        self.mongo.close()
