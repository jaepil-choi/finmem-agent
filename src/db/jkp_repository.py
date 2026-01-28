import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
from src.config import settings

logger = logging.getLogger(__name__)

class JKPRepository:
    """
    Repository for accessing JKP factor return data stored in Parquet format.
    """
    
    def __init__(self, parquet_path: str = "data/jkp-factors/jkp_factors.parquet"):
        self.parquet_path = Path(parquet_path)
        self._df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self):
        """Loads the parquet data into a pandas DataFrame."""
        if not self.parquet_path.exists():
            logger.error(f"JKP Factors parquet file not found at: {self.parquet_path}")
            return
            
        try:
            self._df = pd.read_parquet(self.parquet_path)
            # Ensure date is datetime type
            self._df['date'] = pd.to_datetime(self._df['date'])
            logger.info(f"Loaded JKP data from {self.parquet_path}. Rows: {len(self._df)}")
        except Exception as e:
            logger.error(f"Failed to load JKP data: {e}")

    def get_factor_returns(self, target_date: datetime) -> Dict[str, float]:
        """
        Fetches actual returns for all factors on a given target_date.
        """
        if self._df is None:
            return {}
            
        # Filter for the target date
        day_data = self._df[self._df['date'].dt.date == target_date.date()]
        
        if day_data.empty:
            logger.warning(f"No JKP factor data found for date: {target_date.date()}")
            return {}
            
        # Create a dictionary mapping factor name to return
        returns = {}
        for _, row in day_data.iterrows():
            factor_name = row['name']
            returns[factor_name] = float(row['ret'])
            
        return returns

    def get_factor_history(self, factor_name: str, end_date: datetime, window_days: int = 252) -> pd.Series:
        """
        Fetches historical returns for a specific factor up to end_date.
        Returns a pandas Series indexed by date.
        """
        if self._df is None:
            return pd.Series()

        # Filter by name and date range
        mask = (self._df['name'] == factor_name) & (self._df['date'] <= end_date)
        history = self._df[mask].sort_values('date')

        if history.empty:
            logger.warning(f"No history found for factor: {factor_name} before {end_date}")
            return pd.Series()

        # Tail the last window_days
        history = history.tail(window_days)
        
        # Set date as index and return the 'ret' column as a Series
        return history.set_index('date')['ret']

    def get_available_dates(self) -> List[datetime]:
        """Returns a list of unique dates available in the dataset."""
        if self._df is None:
            return []
        return sorted(self._df['date'].unique().tolist())
