import logging
import sys
import os
import argparse
import numpy as np
from uuid import uuid4
from datetime import datetime, time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from src.core.evaluation.backtester import Backtester
from src.db.jkp_repository import JKPRepository
from src.db.repository import ReportRepository
from src.rag.strategies.finmem import FinMemRAG
from src.config import settings
from src.core.optimization.black_litterman import BlackLittermanOptimizer
from src.graph.builder import create_rag_graph
from src.core.utils.visualizer import get_all_importance_scores, plot_importance_histogram, plot_cumulative_returns
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

# Configure logging to be less noisy for external libs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def reset_all_importance():
    """Resets all importance scores in FAISS to 40.0."""
    print("\n[System] Resetting all document importance scores to 40.0...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    for tier in ["shallow", "intermediate"]:
        tier_path = Path(settings.VECTOR_DB_ROOT) / tier
        if (tier_path / "index.faiss").exists():
            index = FAISS.load_local(str(tier_path), embeddings, allow_dangerous_deserialization=True)
            for doc_id, doc in index.docstore._dict.items():
                doc.metadata["importance"] = 40.0
                doc.metadata["access_counter"] = 0
            index.save_local(str(tier_path))
            print(f"  - Reset {tier} index.")

def run_detailed_training_loop(start_date: datetime, end_date: datetime, is_training: bool = True):
    """
    Runs a detailed training loop with step-by-step logging similar to demo_detailed_factors.py.
    """
    mode_str = "TRAINING (Self-Evolution)" if is_training else "TEST"
    print(f"\n{'='*80}")
    print(f"{f'STARTING {mode_str} LOOP':^80}")
    print(f"{'='*80}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"{'='*80}\n")

    # 1. Initialize Components
    jkp_repo = JKPRepository()
    report_repo = ReportRepository()
    optimizer = BlackLittermanOptimizer()
    rag_strategy = FinMemRAG()
    app = create_rag_graph()
    
    # Define Macro Question
    question = "Predict the directional movement (Up/Neutral/Down) of the target factor for the next period based on the provided daily news and historical context."
    target_factors = "all"
    
    # Mapping for factors
    if target_factors == "all":
        factor_keys = list(settings.factor_expertise.keys())
    else:
        factor_keys = [f.strip() for f in target_factors.split(",")]

    # 2. Get Valid Dates (Intersection of Returns and Reports)
    available_return_dates = {d.date() for d in jkp_repo.get_available_dates()}
    available_report_dates = {d.date() for d in report_repo.get_available_report_dates(collection="daily")}
    valid_dates = sorted(available_return_dates.intersection(available_report_dates))
    
    backtest_dates = [
        datetime.combine(d, time.min) for d in valid_dates 
        if start_date.date() <= d <= end_date.date()
    ]
    
    if not backtest_dates:
        print(f"(!) No valid dates found in range {start_date.date()} to {end_date.date()}")
        return

    cumulative_return = 0.0
    name_map = {v.get('name', k.capitalize()): k for k, v in settings.factor_expertise.items()}
    
    # Tracking for visualization
    tracking_dates = []
    tracking_returns = []
    daily_returns_list = []
    
    # 3. Capture Initial Importance (Training only)
    if is_training:
        reset_all_importance()
        print(f"\n[Visualizer] Capturing initial importance distribution...")
        initial_scores = get_all_importance_scores()
        plot_importance_histogram(initial_scores, "Memory Importance Distribution (Pre-Training)", "plots/importance_pre_training.png")

    # 4. Main Loop
    for current_date in backtest_dates:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n\n{'#'*80}")
        print(f"{f'DATE: {date_str}':^80}")
        print(f"{'#'*80}")

        # --- STEP 1: Fetch actual returns ---
        raw_actual_returns = jkp_repo.get_factor_returns(current_date)
        if not raw_actual_returns:
            print(f"(!) Skipping {date_str}: Missing factor returns.")
            continue
            
        actual_returns = {}
        for raw_name, ret in raw_actual_returns.items():
            internal_key = name_map.get(raw_name)
            if internal_key:
                actual_returns[internal_key] = ret
            else:
                actual_returns[raw_name.lower().replace(" ", "_")] = ret

        # --- STEP 2: Fetch Today's Reports ---
        print(f"\n[Step 1] Fetching Today's Reports...")
        today_reports = report_repo.get_reports_by_date(current_date, collections=["daily"])
        if not today_reports:
            print(f"(!) Warning: No daily news for {date_str}.")
        else:
            print(f"(v) Found {len(today_reports)} daily reports.")

        # --- STEP 3: Retrieve Memory (RAG) ---
        print(f"\n[Step 2] Retrieving Long-term Memory (FinMemRAG)...")
        # We simulate what the graph does to show scores here
        context_docs = rag_strategy.retrieve(query=question, target_date=current_date, k=5)
        for i, doc in enumerate(context_docs):
            scores = doc.metadata.get("finmem_scores", {})
            score_str = f"S:{scores.get('similarity', 0.0):.2f}, R:{scores.get('recency', 0.0):.2f}, I:{scores.get('importance', 0.0):.2f} | Total: {scores.get('composite', 0.0):.2f}"
            print(f"  - Document {i}: {doc.metadata.get('filename', 'Unknown')} ({score_str})")
        print(f"(v) Retrieved {len(context_docs)} relevant documents.")

        # --- STEP 4: Invoke Graph ---
        print(f"\n[Step 3] Executing Agentic Committees & Reflection...")
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "question": question,
            "target_date": current_date,
            "target_factor": target_factors,
            "is_training": is_training,
            "actual_returns": actual_returns,
            "cumulative_return": cumulative_return,
            "messages": []
        }
        
        try:
            # We use invoke to run the whole graph (including reflection if training)
            state_output = app.invoke(initial_state, config=config)
            
            # --- STEP 5: Display Committee Views ---
            committee_results = state_output.get("committee_views", {})
            print(f"\n[Step 4] Committee Results & Votes:")
            for factor_key, view in committee_results.items():
                votes = view.get("individual_votes", [])
                q = view.get("q_value", 0.0)
                omega = view.get("omega_value", 0.0)
                print(f"  - {factor_key.upper():<12} | Q: {q:+.2f} (Bias) | Ω: {omega:.2f} (Uncert) | Votes: {votes}")

            # --- STEP 6: Optimization & Portfolio Return ---
            print(f"\n[Step 5] Portfolio Optimization & Performance:")
            optimized_weights = optimizer.get_optimized_weights(committee_results, factor_keys)
            
            daily_return = 0.0
            print(f"  {'Factor':<15} | {'Weight':<10} | {'Return':<10}")
            print(f"  {'-'*41}")
            for factor, weight in optimized_weights.items():
                if abs(weight) > 0.01:
                    actual_ret = actual_returns.get(factor, 0.0)
                    daily_return += weight * actual_ret
                    print(f"  {factor:<15} | {weight:>10.2%} | {actual_ret:>+10.4f}")
            
            cumulative_return += daily_return
            daily_returns_list.append(daily_return)
            
            print(f"  {'-'*41}")
            print(f"  {'DAILY RETURN':<15} | {daily_return:>+10.4f}")
            print(f"  {'CUMULATIVE':<15} | {cumulative_return:>+10.4f}")
            print(f"  {'PORT RETURNS':<15} | [{', '.join([f'{r:+.4f}' for r in daily_returns_list])}]")

            # Tracking
            tracking_dates.append(current_date)
            tracking_returns.append(cumulative_return)

            # Note: The Reflection logs are printed directly from the nodes

        except Exception as e:
            print(f"(!) Error during training at {date_str}: {e}")
            import traceback
            print(traceback.format_exc())

    print(f"\n\n{'='*80}")
    print(f"{f'{mode_str} LOOP COMPLETE':^80}")
    print(f"{'='*80}")
    print(f"Final Cumulative Return: {cumulative_return:.4f}")
    print(f"{'='*80}\n")
    
    # 5. Final Visualizations
    if is_training:
        print(f"\n[Visualizer] Capturing post-training importance distribution...")
        final_scores = get_all_importance_scores()
        plot_importance_histogram(final_scores, "Memory Importance Distribution (Post-Training)", "plots/importance_post_training.png")
    else:
        # Plot Performance for Test Mode
        print(f"\n[Visualizer] Generating performance plot...")
        plot_cumulative_returns(tracking_dates, tracking_returns, "Strategy Cumulative Performance (Test Period)", "plots/test_performance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Factor-FinMem Training Loop with Detailed Logs")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Execution mode")
    parser.add_argument("--start", type=str, help="Start date (YYYYMMDD)")
    parser.add_argument("--end", type=str, help="End date (YYYYMMDD)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        default_start = "20240225"
        default_end = "20240403"
        is_training = True
    else:
        default_start = "20240404"
        default_end = "20240515"
        is_training = False
        
    start_str = args.start or default_start
    end_str = args.end or default_end
    
    start_dt = datetime.strptime(start_str, "%Y%m%d")
    end_dt = datetime.strptime(end_str, "%Y%m%d")
    
    run_detailed_training_loop(start_dt, end_dt, is_training=is_training)
