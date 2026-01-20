import pandas as pd
from pathlib import Path

def convert_csv_to_parquet(input_csv: str, output_parquet: str):
    """
    Converts a JKP factors CSV to a single optimized Parquet file.
    
    The script ensures proper date typing and lexicographical sorting 
    by (date, name) to optimize for range queries and predicate pushdown.
    """
    print(f"Reading {input_csv}...")
    # Load data
    df = pd.read_csv(input_csv)
    
    # Data Cleaning and Preparation
    print("Formatting data...")
    # Ensure date is datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and theme (name) for optimal Parquet row group statistics
    # This allows efficient slicing by date range or specific factor themes.
    df = df.sort_values(by=['date', 'name']).reset_index(drop=True)
    
    # Save to Parquet
    print(f"Saving to {output_parquet}...")
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Using 'snappy' compression as a standard balance between speed and size
    df.to_parquet(output_parquet, engine='pyarrow', compression='snappy', index=False)
    
    print("Conversion complete.")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(__file__).parent.parent
    INPUT_CSV = BASE_DIR / "data" / "jkp-factors" / "[kor]_[all_themes]_[daily]_[vw_cap].csv"
    OUTPUT_PARQUET = BASE_DIR / "data" / "jkp-factors" / "jkp_factors.parquet"
    
    convert_csv_to_parquet(str(INPUT_CSV), str(OUTPUT_PARQUET))
