import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.rag.strategies.finmem import FinMemRAG

def get_all_importance_scores() -> List[float]:
    """Extracts all importance scores from currently loaded FAISS indices."""
    strategy = FinMemRAG()
    scores = []
    for tier_name, index in strategy.indices.items():
        for internal_id, doc in index.docstore._dict.items():
            importance = doc.metadata.get("importance", 40.0)
            scores.append(float(importance))
    return scores

def plot_importance_histogram(scores: List[float], title: str, save_path: str):
    """Generates and saves a histogram of importance scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=20, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Frequency")
    plt.xlim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")

def plot_cumulative_returns(dates: List[Any], returns: List[float], title: str, save_path: str):
    """Generates and saves a cumulative return curve."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, returns, marker='o', linestyle='-', color='forestgreen', linewidth=2, markersize=4)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")
