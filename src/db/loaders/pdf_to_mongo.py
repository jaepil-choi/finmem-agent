import os
import re
import pdfplumber
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[3]))

from src.db.mongodb_client import MongoDBClient

class KiwoomPDFLoader:
    def __init__(self, data_root="data/Kiwoom"):
        self.data_root = Path(data_root)
        self.mongo = MongoDBClient()
        self.mongo.ensure_timeseries_collections(["daily", "weekly", "monthly"])

    def extract_text(self, pdf_path):
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text

    def parse_date(self, filename, category):
        # 1. Check for YYYYMMDD_ prefix (Common for renamed Monthly/Weekly)
        match_prefix = re.match(r'^(\d{8})_', filename)
        if match_prefix:
            try:
                return datetime.strptime(match_prefix.group(1), "%Y%m%d")
            except ValueError:
                pass

        # 2. Daily & Weekly specific rule: YYYYMMDD or YYMMDD from start + 1 day
        if category in ["daily", "weekly"]:
            # Try 8 digits first
            match8 = re.match(r'^(\d{8})', filename)
            if match8:
                try:
                    dt = datetime.strptime(match8.group(1), "%Y%m%d")
                    from datetime import timedelta
                    return dt + timedelta(days=1)
                except ValueError:
                    pass
            
            # Try 6 digits
            match6 = re.match(r'^(\d{6})', filename)
            if match6:
                try:
                    dt = datetime.strptime(f"20{match6.group(1)}", "%Y%m%d")
                    from datetime import timedelta
                    return dt + timedelta(days=1)
                except ValueError:
                    pass

        # 3. Standard YYYYMMDD anywhere
        match8_any = re.search(r'(\d{8})', filename)
        if match8_any:
            try:
                dt = datetime.strptime(match8_any.group(1), "%Y%m%d")
                if category in ["daily", "weekly"]:
                    from datetime import timedelta
                    return dt + timedelta(days=1)
                return dt
            except ValueError:
                pass
        
        # 4. Search for YYMMDD pattern (6 digits) anywhere
        match6_any = re.search(r'(\d{6})', filename)
        if match6_any:
            try:
                date_str = match6_any.group(1)
                dt = datetime.strptime(f"20{date_str}", "%Y%m%d")
                if category in ["daily", "weekly"]:
                    from datetime import timedelta
                    return dt + timedelta(days=1)
                return dt
            except ValueError:
                pass
        
        return None

    def process_directory(self, category):
        dir_path = self.data_root / category.capitalize()
        if not dir_path.exists():
            print(f"Directory not found: {dir_path}")
            return

        print(f"Processing {category} reports...")
        # Recursively find all pdfs
        pdf_files = list(dir_path.rglob("*.pdf"))
        
        for pdf_path in pdf_files:
            date = self.parse_date(pdf_path.name, category)
            if not date:
                print(f"Skipping {pdf_path.name}: Could not parse date")
                continue

            print(f"  - Loading {pdf_path.name} (Usable at: {date.date()})")
            text = self.extract_text(pdf_path)
            
            if text:
                metadata = {
                    "filename": pdf_path.name,
                    "source_path": str(pdf_path),
                    "category": category,
                    "processed_at": datetime.now()
                }
                self.mongo.insert_report(category, date, pdf_path.name, text, metadata)
                print(f"    Stored in MongoDB.")
            else:
                print(f"    No text extracted from {pdf_path.name}")

    def run(self):
        for category in ["daily", "weekly", "monthly"]:
            self.process_directory(category)

if __name__ == "__main__":
    loader = KiwoomPDFLoader()
    loader.run()
