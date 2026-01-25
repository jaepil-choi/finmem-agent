import os
import re
import pdfplumber
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[3]))

# from src.db.mongodb_client import MongoDBClient

class PDFConverterCLI:
    def __init__(self):
        self.mongo = None

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

    def parse_date(self, filename, category, dir_hint=""):
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
                # If it's a daily/weekly but didn't match the start-of-filename rule
                if category in ["daily", "weekly"]:
                    from datetime import timedelta
                    return dt + timedelta(days=1)
                return dt
            except ValueError:
                pass

        # 4. YY.MM (e.g., 24.5월)
        match_yymm = re.search(r'(\d{2})\.(\d{1,2})월', filename)
        if match_yymm:
            year = int(f"20{match_yymm.group(1)}")
            month = int(match_yymm.group(2))
            return datetime(year, month, 1)

        # 5. YYYYMM (6 digits, e.g., 202204)
        match6 = re.search(r'(\d{6})', filename)
        if match6:
            try:
                val = int(match6.group(1))
                if 200001 <= val <= 210012:
                    return datetime.strptime(match6.group(1), "%Y%m")
                return datetime.strptime(f"20{match6.group(1)}", "%Y%m%d")
            except ValueError:
                pass

        return None

    def get_category(self, pdf_path):
        path_str = str(pdf_path).lower()
        if "daily" in path_str:
            return "daily"
        if "weekly" in path_str:
            return "weekly"
        if "monthly" in path_str:
            return "monthly"
        return "daily" # fallback

    def run(self):
        parser = argparse.ArgumentParser(description="Convert a single PDF to text and optionally store in MongoDB.")
        parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
        parser.add_argument("--save", action="store_true", help="Actually save to MongoDB")
        parser.add_argument("--db", type=str, default="reports", help="MongoDB database name")
        
        args = parser.parse_args()
        pdf_path = Path(args.pdf_path)

        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            return

        print(f"--- Processing PDF: {pdf_path.name} ---")
        
        # 1. Extract Category
        category = self.get_category(pdf_path)
        print(f"Category: {category}")

        # 2. Parse Date
        date = self.parse_date(pdf_path.name, category, dir_hint=str(pdf_path.parent))
        print(f"Parsed Date (Usable at): {date.date() if date else 'Unknown'}")

        # 3. Extract Text
        print("Extracting text...")
        text = self.extract_text(pdf_path)
        
        if not text:
            print("No text could be extracted.")
            return

        # 4. Show Dry-run results
        print("\n--- Text Preview (First 500 chars) ---")
        print(text[:500] + "..." if len(text) > 500 else text)
        print("--------------------------------------")
        print(f"Total Length: {len(text)} characters")

        # 5. Save if requested
        if args.save:
            if not date:
                print("Error: Cannot save to MongoDB without a valid date.")
                return
            
            from src.db.mongodb_client import MongoDBClient
            self.mongo = MongoDBClient(db_name=args.db)
            self.mongo.ensure_timeseries_collections(["daily", "weekly", "monthly"])
            
            metadata = {
                "filename": pdf_path.name,
                "source_path": str(pdf_path),
                "category": category,
                "processed_at": datetime.now(),
                "extraction_method": "pdfplumber"
            }
            
            self.mongo.insert_report(category, date, pdf_path.name, text, metadata)
            print(f"Successfully saved to MongoDB (Collection: {category})")
            self.mongo.close()
        else:
            print("\nDry-run complete. Use --save to store in MongoDB.")

if __name__ == "__main__":
    PDFConverterCLI().run()
