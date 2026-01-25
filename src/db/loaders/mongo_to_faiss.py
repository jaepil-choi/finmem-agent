import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.db.mongodb_client import MongoDBClient

class MongoToFAISSLoader:
    def __init__(self, db_name=None, vector_db_root=None):
        self.db_name = db_name or settings.MONGODB_DB_NAME
        self.vector_db_root = Path(vector_db_root or settings.VECTOR_DB_ROOT)
        self.vector_db_root.mkdir(parents=True, exist_ok=True)
        
        self.mongo = MongoDBClient(db_name=self.db_name)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            add_start_index=True
        )
        
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_tier_for_category(self, category: str) -> str:
        if category == "daily":
            return "shallow"
        if category in ["weekly", "monthly"]:
            return "intermediate"
        return "archive"

    def process_collection(self, collection_name: str, limit: Optional[int] = None, dry_run: bool = False):
        tier = self.get_tier_for_category(collection_name)
        self.logger.info(f"Processing collection '{collection_name}' into tier '{tier}' (dry_run={dry_run}, limit={limit})")
        
        # 1. Fetch documents from MongoDB
        query = self.mongo.db[collection_name].find()
        if limit:
            query = query.limit(limit)
            
        mongo_docs = list(query)
        if not mongo_docs:
            self.logger.info(f"No documents found in collection '{collection_name}'")
            return

        all_chunks = []
        for m_doc in mongo_docs:
            # 2. Convert to LangChain Document
            doc = Document(
                page_content=m_doc["full_text"],
                metadata={
                    "source_id": str(m_doc["_id"]),
                    "date": m_doc["date"].isoformat() if isinstance(m_doc["date"], datetime) else m_doc["date"],
                    "filename": m_doc.get("filename", "unknown"),
                    "category": collection_name,
                    "tier": tier,
                    **m_doc.get("metadata", {})
                }
            )
            
            # 3. Chunking
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        self.logger.info(f"Created {len(all_chunks)} chunks from {len(mongo_docs)} documents")

        if dry_run:
            self.logger.info(f"[Dry-run] Would have saved {len(all_chunks)} chunks to tier '{tier}'")
            if all_chunks:
                sample = all_chunks[0]
                self.logger.info(f"[Dry-run] Sample metadata: {sample.metadata}")
                self.logger.info(f"[Dry-run] Sample content preview: {sample.page_content[:100]}...")
            return

        # 4. Create or Update FAISS index
        tier_path = self.vector_db_root / tier
        
        if (tier_path / "index.faiss").exists():
            self.logger.info(f"Updating existing FAISS index at {tier_path}")
            vectorstore = FAISS.load_local(
                str(tier_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(all_chunks)
        else:
            self.logger.info(f"Creating new FAISS index at {tier_path}")
            vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
        
        # 5. Save locally
        vectorstore.save_local(str(tier_path))
        self.logger.info(f"Successfully saved FAISS index for tier '{tier}'")

    def sync_all(self, limit: Optional[int] = None, dry_run: bool = False):
        for collection in ["daily", "weekly", "monthly"]:
            self.process_collection(collection, limit=limit, dry_run=dry_run)

    def close(self):
        self.mongo.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync MongoDB data to FAISS vector store")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process per collection")
    parser.add_argument("--dry-run", action="store_true", help="Do not save results to FAISS")
    parser.add_argument("--collection", type=str, help="Specific collection to process")
    
    args = parser.parse_args()
    
    loader = MongoToFAISSLoader()
    try:
        if args.collection:
            loader.process_collection(args.collection, limit=args.limit, dry_run=args.dry_run)
        else:
            loader.sync_all(limit=args.limit, dry_run=args.dry_run)
    finally:
        loader.close()
