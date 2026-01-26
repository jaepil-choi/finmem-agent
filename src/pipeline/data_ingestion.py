import argparse
import logging
from src.db.loaders.mongo_loader import KiwoomPDFLoader
from src.db.loaders.faiss_loader import FAISSLoader

def run_ingestion_pipeline(limit=None, dry_run=False, batch_size=50):
    """
    End-to-End Data Ingestion Pipeline:
    1. Ingest PDFs from data/Kiwoom into MongoDB.
    2. Sync ingested MongoDB documents to FAISS Vector DB.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting End-to-End Data Ingestion Pipeline...")
    
    # Step 1: PDF to MongoDB
    logger.info("Step 1: Ingesting PDFs to MongoDB...")
    mongo_loader = KiwoomPDFLoader()
    try:
        # Note: process_all scans the directories and inserts if not already present
        # For simplicity in this version, we run process_all. 
        # Future versions could track last processed file.
        mongo_loader.process_all(save=not dry_run)
    finally:
        mongo_loader.close()
        
    # Step 2: MongoDB to FAISS
    logger.info("Step 2: Syncing MongoDB to FAISS Vector DB...")
    faiss_loader = FAISSLoader()
    try:
        faiss_loader.sync_all(limit=limit, dry_run=dry_run, batch_size=batch_size)
    finally:
        faiss_loader.close()
        
    logger.info("E2E Data Ingestion Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Data Ingestion Pipeline")
    parser.add_argument("--limit", type=int, help="Limit number of documents to sync to FAISS per collection")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving to databases")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding requests")
    
    args = parser.parse_args()
    
    run_ingestion_pipeline(
        limit=args.limit, 
        dry_run=args.dry_run, 
        batch_size=args.batch_size
    )
