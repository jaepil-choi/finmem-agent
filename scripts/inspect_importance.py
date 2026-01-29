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

def inspect_index(tier="shallow"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    tier_path = Path(settings.VECTOR_DB_ROOT) / tier
    print(f"Loading index from {tier_path}")
    index = FAISS.load_local(str(tier_path), embeddings, allow_dangerous_deserialization=True)
    
    # Get first 5 documents from the docstore
    doc_ids = list(index.docstore._dict.keys())[:5]
    print(f"\nSample metadata from {tier} index:")
    for doc_id in doc_ids:
        doc = index.docstore._dict[doc_id]
        print(f"ID: {doc_id}")
        print(f"Metadata: {doc.metadata}")
        # print(f"Content preview: {doc.page_content[:50]}...")
        print("-" * 20)

if __name__ == "__main__":
    inspect_index("shallow")
    inspect_index("intermediate")
