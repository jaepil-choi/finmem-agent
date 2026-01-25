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

    def parse_date(self, filename):
        # 1. Search for YYYYMMDD pattern (8 digits)
        match8 = re.search(r'(\d{8})', filename)
        if match8:
            try:
                return datetime.strptime(match8.group(1), "%Y%m%d")
            except ValueError:
                pass
        
        # 2. Search for YYMMDD pattern (6 digits) at the beginning or after an underscore
        match6 = re.search(r'(?:^|_| )(\d{6})(?:_|\.|$| )', filename)
        if match6:
            try:
                date_str = match6.group(1)
                # Assume 20xx for 2-digit year
                return datetime.strptime(f"20{date_str}", "%Y%m%d")
            except ValueError:
                pass
        
        # 3. Fallback: just look for 6 digits anywhere if others fail
        match6_any = re.search(r'(\d{6})', filename)
        if match6_any:
            try:
                date_str = match6_any.group(1)
                return datetime.strptime(f"20{date_str}", "%Y%m%d")
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
            date = self.parse_date(pdf_path.name)
            if not date:
                print(f"Skipping {pdf_path.name}: Could not parse date")
                continue

            print(f"  - Loading {pdf_path.name} ({date.date()})")
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
