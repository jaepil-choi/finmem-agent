import re
import pdfplumber
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.config import settings
from src.db.mongodb_client import MongoDBClient

class KiwoomPDFLoader:
    def __init__(self, data_root="data/Kiwoom", db_name=None):
        self.data_root = Path(data_root)
        self.db_name = db_name or settings.MONGODB_DB_NAME
        self._mongo = None
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @property
    def mongo(self):
        if self._mongo is None:
            self._mongo = MongoDBClient(db_name=self.db_name)
            self._mongo.ensure_timeseries_collections(["daily", "weekly", "monthly"])
        return self._mongo

    def extract_text(self, pdf_path):
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
        return text

    def parse_date(self, filename, category):
        dt = None
        # 1. Check for YYYYMMDD_ prefix
        match_prefix = re.match(r'^(\d{8})_', filename)
        if match_prefix:
            try:
                dt = datetime.strptime(match_prefix.group(1), "%Y%m%d")
            except ValueError:
                pass

        # 2. Daily & Weekly specific rule: YYYYMMDD or YYMMDD from start
        if not dt and category in ["daily", "weekly"]:
            match8 = re.match(r'^(\d{8})', filename)
            if match8:
                try:
                    dt = datetime.strptime(match8.group(1), "%Y%m%d")
                except ValueError:
                    pass
            
            if not dt:
                match6 = re.match(r'^(\d{6})', filename)
                if match6:
                    try:
                        dt = datetime.strptime(f"20{match6.group(1)}", "%Y%m%d")
                    except ValueError:
                        pass

        # 3. Standard YYYYMMDD anywhere
        if not dt:
            match8_any = re.search(r'(\d{8})', filename)
            if match8_any:
                try:
                    dt = datetime.strptime(match8_any.group(1), "%Y%m%d")
                except ValueError:
                    pass
        
        # 4. Search for YYMMDD pattern (6 digits) anywhere
        if not dt:
            match6_any = re.search(r'(?:^|_| )(\d{6})(?:_|\.|$| )', filename)
            if match6_any:
                try:
                    date_str = match6_any.group(1)
                    dt = datetime.strptime(f"20{date_str}", "%Y%m%d")
                except ValueError:
                    pass
        
        # Apply +1 day lag for Daily and Weekly to avoid forward-looking bias
        if dt and category in ["daily", "weekly"]:
            dt = dt + timedelta(days=1)
            
        return dt

    def get_category(self, pdf_path):
        path_str = str(pdf_path).lower()
        if "daily" in path_str: return "daily"
        if "weekly" in path_str: return "weekly"
        if "monthly" in path_str: return "monthly"
        return "daily"

    def process_file(self, pdf_path, save=True):
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.warning(f"File not found: {pdf_path}")
            return False

        category = self.get_category(pdf_path)
        date = self.parse_date(pdf_path.name, category)
        
        if not date:
            self.logger.warning(f"Skipping {pdf_path.name}: Could not parse date")
            return False

        self.logger.info(f"Processing {pdf_path.name} (Category: {category}, Usable at: {date.date()})")
        text = self.extract_text(pdf_path)
        
        if not text:
            self.logger.warning(f"  No text extracted from {pdf_path.name}")
            return False

        if save:
            metadata = {
                "filename": pdf_path.name,
                "source_path": str(pdf_path),
                "category": category,
                "processed_at": datetime.now(),
                "extraction_method": "pdfplumber"
            }
            self.mongo.insert_report(category, date, pdf_path.name, text, metadata)
            self.logger.info(f"  Successfully saved to MongoDB.")
        else:
            self.logger.info(f"  [Dry-run] Text Length: {len(text)} chars")
            self.logger.info(f"  [Dry-run] Preview: {text[:200].replace('\n', ' ')}...")
        
        return True

    def process_all(self, save=True):
        for category in ["daily", "weekly", "monthly"]:
            dir_path = self.data_root / category.capitalize()
            if not dir_path.exists(): continue
            
            self.logger.info(f"--- Batch processing {category} reports ---")
            pdf_files = list(dir_path.rglob("*.pdf"))
            for pdf_path in pdf_files:
                self.process_file(pdf_path, save=save)

    def close(self):
        if self._mongo:
            self._mongo.close()

if __name__ == "__main__":
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
