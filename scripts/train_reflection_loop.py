import logging
import sys
import os
from datetime import datetime
from src.core.evaluation.backtester import Backtester
from src.db.jkp_repository import JKPRepository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_backtest():
    """
    Runs a multi-day backtest in training mode to allow the system 
    to evolve its memory based on market feedback.
    """
    logger.info("=== Starting Self-Evolution Training Loop ===")
    
    # 1. Define Date Range
    # AI/Semiconductor boom was prominent in late 2023
    start_date = datetime(2023, 11, 1)
    end_date = datetime(2023, 11, 10) # 10 days for testing
    
    # 2. Define Macro Question
    # This question will be used by all committees to focus their analysis
    question = "Analyze the impact of the current AI and high-performance computing (HPC) demand on factor performance. Should we overweight low-leverage or quality stocks?"
    
    # 3. Initialize Backtester
    backtester = Backtester(start_date=start_date, end_date=end_date)
    
    # 4. Run Backtest in Training Mode
    # We focus on a few key factors for the test to save tokens/time
    # or use 'all' for full coverage
    target_factors = "low_leverage, size, value" 
    
    logger.info(f"Target Factors: {target_factors}")
    logger.info(f"Question: {question}")
    
    backtester.run(question=question, target_factor=target_factors, is_training=True)
    
    # 5. Display Summary
    summary = backtester.get_summary()
    if not summary.empty:
        print("\n=== Backtest Training Summary ===")
        print(summary)
        print(f"\nFinal Cumulative Return: {summary['cumulative_return'].iloc[-1]:.4f}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    # Ensure environment variables are loaded
    # OPENAI_API_KEY must be set
    try:
        run_training_backtest()
    except Exception as e:
        logger.error(f"Failed to run training loop: {e}")
        import traceback
        logger.error(traceback.format_exc())
