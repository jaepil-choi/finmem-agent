import logging
import sys
import os
import argparse
from datetime import datetime
from src.core.evaluation.backtester import Backtester
from src.db.jkp_repository import JKPRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_backtest(start_date: datetime, end_date: datetime, is_training: bool = True):
    """
    Runs a multi-day backtest in training or test mode.
    """
    mode_str = "TRAINING (Self-Evolution)" if is_training else "TEST"
    logger.info(f"=== Starting {mode_str} Loop ===")
    
    # Define Macro Question
    question = "Predict the directional movement (Up/Neutral/Down) of the target factor for the next period based on the provided daily news and historical context."
    
    # Initialize Backtester
    backtester = Backtester(start_date=start_date, end_date=end_date)
    
    # Run Backtest
    # Using 'all' to cover all 13 factors as requested
    target_factors = "all" 
    
    logger.info(f"Target Factors: {target_factors}")
    logger.info(f"Question: {question}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    
    backtester.run(question=question, target_factor=target_factors, is_training=is_training)
    
    # Display Summary
    summary = backtester.get_summary()
    if not summary.empty:
        print(f"\n=== Backtest {mode_str} Summary ===")
        print(summary)
        if 'cumulative_return' in summary.columns:
            print(f"\nFinal Cumulative Return: {summary['cumulative_return'].iloc[-1]:.4f}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Factor-FinMem Training or Test Loop")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Execution mode")
    parser.add_argument("--start", type=str, help="Start date (YYYYMMDD)")
    parser.add_argument("--end", type=str, help="End date (YYYYMMDD)")
    
    args = parser.parse_args()
    
    # Default Dates based on request
    # Training: 20240225 to 20240403
    # Test: 20240404 to 20240515
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
    
    try:
        run_training_backtest(start_dt, end_dt, is_training=is_training)
    except Exception as e:
        logger.error(f"Failed to run backtest loop: {e}")
        import traceback
        logger.error(traceback.format_exc())
