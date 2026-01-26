import argparse
from src.db.loaders.mongo_loader import KiwoomPDFLoader

def main():
    parser = argparse.ArgumentParser(description="Kiwoom PDF to MongoDB Loader")
    parser.add_argument("--file", type=str, help="Path to a specific PDF file to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not save to database, just show results")
    
    args = parser.parse_args()
    loader = KiwoomPDFLoader()
    
    try:
        if args.file:
            loader.process_file(args.file, save=not args.dry_run)
        else:
            loader.process_all(save=not args.dry_run)
    finally:
        loader.close()

if __name__ == "__main__":
    main()
