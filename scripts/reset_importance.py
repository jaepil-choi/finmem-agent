import sys
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())
load_dotenv()

from src.config import settings

def reset_importance(tier="shallow"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    tier_path = Path(settings.VECTOR_DB_ROOT) / tier
    if not (tier_path / "index.faiss").exists():
        print(f"Skipping {tier}, index not found.")
        return

    print(f"Loading index from {tier_path}")
    index = FAISS.load_local(str(tier_path), embeddings, allow_dangerous_deserialization=True)
    
    count = 0
    for doc_id, doc in index.docstore._dict.items():
        doc.metadata["importance"] = 40.0
        doc.metadata["access_counter"] = 0
        count += 1
    
    print(f"Reset {count} documents in {tier} index. Saving...")
    index.save_local(str(tier_path))
    print(f"Successfully reset and saved {tier} index.")

if __name__ == "__main__":
    reset_importance("shallow")
    reset_importance("intermediate")
