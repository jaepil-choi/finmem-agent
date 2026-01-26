import argparse
from src.db.loaders.faiss_loader import FAISSLoader

def main():
    parser = argparse.ArgumentParser(description="Sync MongoDB data to FAISS vector store")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process per collection")
    parser.add_argument("--dry-run", action="store_true", help="Do not save results to FAISS")
    parser.add_argument("--collection", type=str, help="Specific collection to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding requests")
    
    args = parser.parse_args()
    loader = FAISSLoader()
    
    try:
        if args.collection:
            loader.process_collection(args.collection, limit=args.limit, dry_run=args.dry_run, batch_size=args.batch_size)
        else:
            loader.sync_all(limit=args.limit, dry_run=args.dry_run, batch_size=args.batch_size)
    finally:
        loader.close()

if __name__ == "__main__":
    main()
