import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from src.graph.builder import create_rag_graph
from src.db.jkp_repository import JKPRepository
from src.core.optimization.black_litterman import BlackLittermanOptimizer
from src.config import settings

logger = logging.getLogger(__name__)

class Backtester:
    """
    Orchestrates the backtesting process across a range of dates, 
    integrating the RAG graph, factor returns, and optimization.
    """
    
    def __init__(
        self, 
        start_date: datetime, 
        end_date: datetime,
        jkp_repo: Optional[JKPRepository] = None,
        optimizer: Optional[BlackLittermanOptimizer] = None
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.jkp_repo = jkp_repo or JKPRepository()
        self.optimizer = optimizer or BlackLittermanOptimizer()
        self.app = create_rag_graph()
        self.results: List[Dict[str, Any]] = []

    def run(self, question: str, target_factor: str = "all", is_training: bool = True):
        """
        Runs the backtest loop.
        """
        available_dates = self.jkp_repo.get_available_dates()
        # Filter for dates in range
        backtest_dates = [
            d for d in available_dates 
            if self.start_date <= d <= self.end_date
        ]
        
        if not backtest_dates:
            logger.warning(f"No available JKP data found between {self.start_date} and {self.end_date}")
            return
            
        logger.info(f"Starting backtest for {len(backtest_dates)} days...")
        
        cumulative_return = 0.0
        
        for current_date in backtest_dates:
            logger.info(f"--- Backtest Date: {current_date.date()} ---")
            
            # Fetch actual returns for reflection and performance tracking
            actual_returns = self.jkp_repo.get_factor_returns(current_date)
            
            # Pick a representative actual_return for the state (e.g., the first requested factor)
            # This is used by the single-factor reflection node logic for now
            if target_factor == "all":
                rep_factor = list(settings.factor_expertise.keys())[0]
            else:
                rep_factor = target_factor.split(",")[0].strip()
            
            rep_actual_return = actual_returns.get(rep_factor, 0.0)
            
            # 1. Invoke the LangGraph StateGraph
            thread_id = str(uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            
            initial_state = {
                "question": question,
                "target_date": current_date,
                "target_factor": target_factor,
                "is_training": is_training,
                "actual_return": rep_actual_return,
                "cumulative_return": cumulative_return,
                "messages": []
            }
            
            try:
                state_output = self.app.invoke(initial_state, config=config)
                
                # 2. Extract Committee Views and Optimize
                committee_results = state_output.get("committee_views", {})
                
                # Factors to optimize (either all or the specified subset)
                if target_factor == "all":
                    opt_factors = list(settings.factor_expertise.keys())
                else:
                    opt_factors = [f.strip() for f in target_factor.split(",")]
                    
                optimized_weights = self.optimizer.get_optimized_weights(
                    committee_results, 
                    opt_factors
                )
                
                # 3. Calculate Daily Portfolio Return
                daily_return = 0.0
                for factor, weight in optimized_weights.items():
                    actual_ret = actual_returns.get(factor, 0.0)
                    daily_return += weight * actual_ret
                
                cumulative_return += daily_return
                
                # 4. Record Results
                self.results.append({
                    "date": current_date,
                    "daily_return": daily_return,
                    "cumulative_return": cumulative_return,
                    "weights": optimized_weights,
                    "committee_views": committee_results,
                    "reflections": state_output.get("reflections", {})
                })
                
                logger.info(f"Daily Return: {daily_return:.4f}, Cumulative: {cumulative_return:.4f}")
                
            except Exception as e:
                logger.error(f"Error during backtest at {current_date}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def get_summary(self) -> pd.DataFrame:
        """Returns the backtest results as a DataFrame."""
        if not self.results:
            return pd.DataFrame()
            
        summary_data = []
        for r in self.results:
            summary_data.append({
                "date": r["date"],
                "daily_return": r["daily_return"],
                "cumulative_return": r["cumulative_return"]
            })
        return pd.DataFrame(summary_data)
