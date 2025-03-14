#!/usr/bin/env python3
"""
Timestamp Debug Script

This script diagnoses the timestamp comparison issue by printing detailed information
about the data types and values involved.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample data for testing
SAMPLE_CSV = "top_election_markets.csv"
TRADES_DIR = "/Users/helenwu/Desktop/ML/subgraph/polymarket_raw_data/trades"

def load_sample_data():
    """Load a few rows from the CSV to examine date formats."""
    try:
        df = pd.read_csv(SAMPLE_CSV)
        return df.head(5)
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return None

def examine_date_columns(df):
    """Examine the date columns in detail."""
    if df is None or df.empty:
        logger.error("No data to examine")
        return
    
    date_columns = ['startDate', 'endDate', 'createdAt', 'updatedAt']
    
    for col in date_columns:
        if col in df.columns:
            logger.info(f"\nExamining column: {col}")
            
            # Show raw values
            logger.info(f"Raw values: {df[col].tolist()}")
            
            # Show data types
            logger.info(f"Column dtype: {df[col].dtype}")
            
            # Try different parsing methods
            logger.info("Testing different parse methods:")
            
            for idx, val in enumerate(df[col]):
                if pd.isna(val):
                    continue
                    
                logger.info(f"  Value {idx}: '{val}' (type: {type(val)})")
                
                # Method 1: pd.to_datetime
                try:
                    dt1 = pd.to_datetime(val)
                    logger.info(f"    pd.to_datetime: {dt1} (type: {type(dt1)})")
                except Exception as e:
                    logger.info(f"    pd.to_datetime failed: {e}")
                
                # Method 2: pd.to_datetime with utc=True
                try:
                    dt2 = pd.to_datetime(val, utc=True)
                    logger.info(f"    pd.to_datetime(utc=True): {dt2} (type: {type(dt2)})")
                except Exception as e:
                    logger.info(f"    pd.to_datetime(utc=True) failed: {e}")
                
                # Method 3: Replace Z and parse
                try:
                    fixed_val = val.replace('Z', '+00:00') if isinstance(val, str) and 'Z' in val else val
                    dt3 = pd.to_datetime(fixed_val)
                    logger.info(f"    pd.to_datetime(fixed_val): {dt3} (type: {type(dt3)})")
                except Exception as e:
                    logger.info(f"    pd.to_datetime(fixed_val) failed: {e}")
                
                # Method 4: Convert to timestamp
                try:
                    dt4 = pd.to_datetime(val).timestamp()
                    logger.info(f"    timestamp: {dt4} (type: {type(dt4)})")
                except Exception as e:
                    logger.info(f"    timestamp conversion failed: {e}")

def load_sample_trades():
    """Load a sample trade file to examine datetime values."""
    for filename in os.listdir(TRADES_DIR):
        if filename.endswith('.parquet'):
            try:
                file_path = os.path.join(TRADES_DIR, filename)
                logger.info(f"\nLoading sample trade file: {filename}")
                trades_df = pd.read_parquet(file_path)
                
                if trades_df.empty:
                    logger.info("  Trade file is empty")
                    continue
                
                # Check columns
                logger.info(f"  Columns: {trades_df.columns.tolist()}")
                
                # Examine timestamp column
                if 'timestamp' in trades_df.columns:
                    logger.info(f"  Timestamp column dtype: {trades_df['timestamp'].dtype}")
                    logger.info(f"  First few timestamp values: {trades_df['timestamp'].head().tolist()}")
                    
                    # Check if timestamps are integers
                    if pd.api.types.is_integer_dtype(trades_df['timestamp']):
                        logger.info("  Timestamps appear to be Unix timestamps (integers)")
                    else:
                        logger.info("  Timestamps are not integers")
                    
                    # Try converting to datetime
                    try:
                        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='s')
                        logger.info(f"  Converted datetime column dtype: {trades_df['datetime'].dtype}")
                        logger.info(f"  First few datetime values: {trades_df['datetime'].head().tolist()}")
                    except Exception as e:
                        logger.info(f"  Error converting timestamps to datetime: {e}")
                
                # Only process one file for brevity
                break
                
            except Exception as e:
                logger.error(f"Error loading trade file {filename}: {e}")
                continue

def simulate_comparison():
    """Simulate the comparison that's failing in the main script."""
    logger.info("\nSimulating datetime comparisons:")
    
    # Create dummy dataframe with datetime column
    trades_df = pd.DataFrame({
        'timestamp': [1640995200, 1641081600, 1641168000],  # Unix timestamps
        'datetime': pd.to_datetime([1640995200, 1641081600, 1641168000], unit='s')
    })
    
    # Create different types of end dates
    end_formats = {
        'datetime64': pd.to_datetime('2024-11-05T12:00:00Z'),
        'timestamp': pd.to_datetime('2024-11-05T12:00:00Z').timestamp(),
        'string': '2024-11-05T12:00:00Z'
    }
    
    for name, end_date in end_formats.items():
        logger.info(f"\nTesting end_date as {name}: {end_date} (type: {type(end_date)})")
        
        # Try comparison directly
        try:
            if isinstance(end_date, str):
                end_datetime = pd.to_datetime(end_date.replace('Z', '+00:00'))
            elif isinstance(end_date, (int, float)):
                end_datetime = pd.to_datetime(end_date, unit='s')
            else:
                end_datetime = end_date
                
            logger.info(f"  Converted to datetime: {end_datetime} (type: {type(end_datetime)})")
            
            # Simulate the comparison
            filtered = trades_df[trades_df['datetime'] <= end_datetime]
            logger.info(f"  Comparison works, got {len(filtered)} rows")
        except Exception as e:
            logger.error(f"  Comparison failed: {e}")

def main():
    """Run diagnostics to understand the timestamp comparison issue."""
    logger.info("Starting timestamp diagnostics")
    
    # Examine sample data from the CSV
    df = load_sample_data()
    examine_date_columns(df)
    
    # Examine sample trade data
    load_sample_trades()
    
    # Simulate the comparison
    simulate_comparison()
    
    logger.info("\nDiagnostics complete")

if __name__ == "__main__":
    main()