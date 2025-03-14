#!/usr/bin/env python3
"""
Comprehensive Market Analysis Generator (Fixed Version 3)

This script creates a comprehensive CSV file containing market analysis metrics
by combining original CSV data with calculated metrics from trade data stored in
parquet files.
"""

import os
import sys
import json
import ast
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_analysis_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 8  # Number of parallel workers
TRADES_DIR = "polymarket_raw_data/trades"
OUTPUT_DIR = "analysis_results"

def parse_token_ids(token_ids_str):
    """Parse token IDs from string representation in CSV."""
    try:
        if not token_ids_str or pd.isna(token_ids_str):
            return []
            
        # Handle different string formats
        if isinstance(token_ids_str, str):
            if token_ids_str.startswith('[') and token_ids_str.endswith(']'):
                try:
                    return json.loads(token_ids_str.replace("'", '"'))
                except:
                    return ast.literal_eval(token_ids_str)
            else:
                return token_ids_str.split(',')
        return []
    except Exception as e:
        logger.warning(f"Error parsing token IDs: {e}")
        return []

def load_trade_data(token_id):
    """Load trade data for a specific token from parquet file."""
    try:
        # Attempt to sanitize token_id for file system
        safe_token_id = token_id.replace('/', '_').replace('\\', '_')
        
        file_path = os.path.join(TRADES_DIR, f"{safe_token_id}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        
        # Try with just the token ID without .parquet extension
        file_path = os.path.join(TRADES_DIR, f"{safe_token_id}")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
            
        logger.warning(f"No trade data file found for token {token_id}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading trade data for token {token_id}: {e}")
        return pd.DataFrame()

def calculate_market_metrics(row, all_metrics=True):
    """
    Calculate comprehensive metrics for a market.
    
    Args:
        row: DataFrame row containing market data
        all_metrics: Whether to calculate all metrics or just basic ones
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {}
    
    try:
        # Parse dates
        start_date = None
        end_date = None
        
        if 'startDate' in row and not pd.isna(row['startDate']):
            try:
                start_date_str = row['startDate'].replace('Z', '+00:00') if isinstance(row['startDate'], str) else row['startDate']
                start_date = pd.to_datetime(start_date_str)
            except Exception as e:
                logger.warning(f"Could not parse startDate: {row['startDate']} - {e}")
        
        if 'endDate' in row and not pd.isna(row['endDate']):
            try:
                end_date_str = row['endDate'].replace('Z', '+00:00') if isinstance(row['endDate'], str) else row['endDate']
                end_date = pd.to_datetime(end_date_str)
            except Exception as e:
                logger.warning(f"Could not parse endDate: {row['endDate']} - {e}")
        
        # 1. Days Active
        if start_date is not None and end_date is not None:
            try:
                delta = end_date - start_date
                metrics['days_active'] = max(1, delta.days)  # Ensure at least 1 day
            except Exception as e:
                logger.warning(f"Could not calculate days_active: {e}")
        
        # Parse token IDs
        token_ids = parse_token_ids(row['clobTokenIds']) if 'clobTokenIds' in row else []
        if not token_ids:
            # Return basic metrics if no token IDs available
            return metrics
        
        # Initialize aggregates
        all_trades_df = pd.DataFrame()
        trader_sets = []
        
        # Process each token for this market
        for token_id in token_ids:
            trades_df = load_trade_data(token_id)
            if trades_df.empty:
                continue
                
            # Convert timestamp to datetime if needed
            if 'timestamp' in trades_df.columns:
                try:
                    # Convert timestamps to numeric before conversion
                    trades_df['timestamp_num'] = pd.to_numeric(trades_df['timestamp'], errors='coerce')
                    trades_df['datetime'] = pd.to_datetime(trades_df['timestamp_num'], unit='s')
                except Exception as e:
                    logger.warning(f"Error converting timestamps to datetime for token {token_id}: {e}")
                    continue
            
            # Add token info
            trades_df['token_id'] = token_id
            
            # Collect unique traders for this token
            if 'maker_id' in trades_df.columns and 'taker_id' in trades_df.columns:
                trader_set = set(trades_df['maker_id'].unique()).union(set(trades_df['taker_id'].unique()))
                trader_sets.append(trader_set)
            
            # Add to combined dataframe
            all_trades_df = pd.concat([all_trades_df, trades_df])
        
        # Return basic metrics if no trade data available
        if all_trades_df.empty:
            return metrics
        
        # 2. Total Volume
        if 'size' in all_trades_df.columns:
            all_trades_df['size'] = pd.to_numeric(all_trades_df['size'], errors='coerce')
            metrics['total_volume'] = all_trades_df['size'].sum()
        
        # 3. Average Daily Volume
        if metrics.get('days_active') and metrics.get('total_volume'):
            metrics['avg_daily_volume'] = metrics['total_volume'] / metrics['days_active']
        
        # Calculate unique traders across all tokens
        all_traders = set()
        for trader_set in trader_sets:
            all_traders.update(trader_set)
        
        # 4. Trader Count
        metrics['trader_count'] = len(all_traders)
        
        # 5. Trading Frequency
        if metrics.get('trader_count') and not all_trades_df.empty:
            metrics['trading_frequency'] = len(all_trades_df) / metrics['trader_count']
        
        # 6. Buy/Sell Ratio
        if 'side' in all_trades_df.columns and 'size' in all_trades_df.columns:
            buy_volume = all_trades_df[all_trades_df['side'] == 'Buy']['size'].sum()
            sell_volume = all_trades_df[all_trades_df['side'] == 'Sell']['size'].sum()
            if sell_volume > 0:
                metrics['buy_sell_ratio'] = buy_volume / sell_volume
            else:
                metrics['buy_sell_ratio'] = buy_volume if buy_volume > 0 else 0
        
        # If not calculating all metrics, return what we have so far
        if not all_metrics:
            return metrics
        
        # Calculate more advanced metrics if end_date is available
        if end_date is not None and 'datetime' in all_trades_df.columns:
            try:
                # Convert end_date to numpy timestamp for correct comparison
                end_date_timestamp = np.datetime64(end_date) 
                
                # Calculate time windows (also as numpy timestamps for consistency)
                cutoff_time_48h = np.datetime64(end_date - timedelta(hours=48))
                cutoff_time_week = np.datetime64(end_date - timedelta(days=7))
                
                # Convert trades_df datetime to numpy datetime64
                if 'datetime' in all_trades_df.columns:
                    all_trades_df['datetime64'] = all_trades_df['datetime'].astype('datetime64[ns]')
                
                # 7. Pre-Election Price (Last trade price)
                pre_election_trades = all_trades_df[all_trades_df['datetime64'] <= end_date_timestamp].sort_values('datetime64')
                if not pre_election_trades.empty and 'price' in pre_election_trades.columns:
                    pre_election_trades['price'] = pd.to_numeric(pre_election_trades['price'], errors='coerce')
                    metrics['pre_election_price'] = float(pre_election_trades.iloc[-1]['price'])
                    
                    # 8. Pre-Election VWAP (48 hours)
                    final_window = pre_election_trades[pre_election_trades['datetime64'] >= cutoff_time_48h]
                    if not final_window.empty and 'price' in final_window.columns and 'size' in final_window.columns:
                        final_window['value'] = final_window['price'] * final_window['size']
                        vwap = final_window['value'].sum() / final_window['size'].sum()
                        metrics['pre_election_vwap'] = vwap
                        
                        # Calculate volume in final 48 hours
                        metrics['final_48h_volume'] = final_window['size'].sum()
                    
                    # 9. Price Volatility (coefficient of variation in final week)
                    final_week = pre_election_trades[pre_election_trades['datetime64'] >= cutoff_time_week]
                    if not final_week.empty and 'price' in final_week.columns:
                        price_std = final_week['price'].std()
                        price_mean = final_week['price'].mean()
                        if price_mean > 0:
                            metrics['price_volatility'] = price_std / price_mean
                    
                    # Calculate price fluctuations (crossings of 50% threshold)
                    if not final_week.empty and 'price' in final_week.columns and len(final_week) >= 2:
                        prices = final_week.sort_values('datetime64')['price'].values
                        crossings = 0
                        for i in range(1, len(prices)):
                            if (prices[i-1] < 0.5 and prices[i] >= 0.5) or (prices[i-1] >= 0.5 and prices[i] < 0.5):
                                crossings += 1
                        metrics['price_fluctuations'] = crossings
                    
                    # Calculate final week momentum
                    if not final_week.empty and 'price' in final_week.columns and len(final_week) >= 2:
                        sorted_week = final_week.sort_values('datetime64')
                        first_price = sorted_week.iloc[0]['price']
                        last_price = sorted_week.iloc[-1]['price']
                        metrics['final_week_momentum'] = float(last_price) - float(first_price)
                
                # Calculate volume acceleration (ratio of final week volume to average weekly volume)
                final_week = all_trades_df[all_trades_df['datetime64'] >= cutoff_time_week]
                if not final_week.empty and 'size' in final_week.columns and metrics.get('days_active'):
                    final_week_volume = final_week['size'].sum()
                    avg_weekly_volume = metrics['total_volume'] / (metrics['days_active'] / 7)
                    if avg_weekly_volume > 0:
                        metrics['volume_acceleration'] = final_week_volume / avg_weekly_volume
                
                # Calculate late-stage participation
                if not final_week.empty and 'size' in final_week.columns and metrics.get('total_volume'):
                    final_week_volume = final_week['size'].sum()
                    metrics['late_stage_participation'] = final_week_volume / metrics['total_volume']
            
            except Exception as e:
                logger.warning(f"Error calculating time-based metrics: {e}")
        
        # Calculate two-way traders
        try:
            if 'maker_id' in all_trades_df.columns and 'taker_id' in all_trades_df.columns and 'side' in all_trades_df.columns:
                buyers = set()
                sellers = set()
                
                # Find all buyers
                buys = all_trades_df[all_trades_df['side'] == 'Buy']
                for _, trade in buys.iterrows():
                    if pd.notna(trade['taker_id']):
                        buyers.add(trade['taker_id'])
                
                # Find all sellers
                sells = all_trades_df[all_trades_df['side'] == 'Sell']
                for _, trade in sells.iterrows():
                    if pd.notna(trade['taker_id']):
                        sellers.add(trade['taker_id'])
                
                # Calculate two-way traders ratio
                two_way_traders = buyers.intersection(sellers)
                if metrics.get('trader_count', 0) > 0:
                    metrics['two_way_traders_ratio'] = len(two_way_traders) / metrics['trader_count']
        except Exception as e:
            logger.warning(f"Error calculating two-way traders: {e}")
        
        # Calculate volume concentration
        try:
            if 'maker_id' in all_trades_df.columns and 'taker_id' in all_trades_df.columns and 'size' in all_trades_df.columns:
                trader_volumes = {}
                
                # Calculate volume per trader
                for _, trade in all_trades_df.iterrows():
                    size = float(trade['size'])
                    
                    # Add to maker volume
                    if pd.notna(trade['maker_id']):
                        trader_id = trade['maker_id']
                        if trader_id not in trader_volumes:
                            trader_volumes[trader_id] = 0
                        trader_volumes[trader_id] += size / 2  # Split volume between maker and taker
                    
                    # Add to taker volume
                    if pd.notna(trade['taker_id']):
                        trader_id = trade['taker_id']
                        if trader_id not in trader_volumes:
                            trader_volumes[trader_id] = 0
                        trader_volumes[trader_id] += size / 2  # Split volume between maker and taker
                
                # Calculate concentration
                if trader_volumes:
                    volumes = sorted(trader_volumes.values(), reverse=True)
                    total_volume = sum(volumes)
                    if total_volume > 0:
                        top_10pct_count = max(1, int(len(volumes) * 0.1))
                        top_10pct_volume = sum(volumes[:top_10pct_count])
                        metrics['volume_concentration'] = top_10pct_volume / total_volume
        except Exception as e:
            logger.warning(f"Error calculating volume concentration: {e}")
        
        # Calculate trader-to-volume ratio
        if metrics.get('total_volume') and metrics.get('trader_count', 0) > 0:
            metrics['trader_to_volume_ratio'] = metrics['total_volume'] / metrics['trader_count']
        
        # Count number of outcomes from the CSV data
        if not pd.isna(row.get('outcomes')):
            try:
                outcomes = parse_token_ids(row['outcomes'])
                metrics['outcome_count'] = len(outcomes)
            except Exception as e:
                logger.warning(f"Error parsing outcomes: {e}")
    
    except Exception as e:
        logger.warning(f"Error in calculate_market_metrics: {e}")
    
    return metrics

def process_market(row, include_columns):
    """Process a single market row to generate comprehensive metrics."""
    try:
        # Create output dictionary with original fields
        result = {col: row[col] for col in include_columns if col in row}
        
        # Calculate metrics
        metrics = calculate_market_metrics(row)
        
        # Combine with original data
        result.update(metrics)
        
        return result
    except Exception as e:
        import traceback
        logger.error(f"Error processing market {row.get('id', 'unknown')}: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        
        # Return basic data without calculated metrics
        return {col: row[col] for col in include_columns if col in row}

def main():
    # Global declaration before use
    global TRADES_DIR
    
    parser = argparse.ArgumentParser(description="Comprehensive Market Analysis Generator")
    parser.add_argument("--input", required=True, help="Input CSV file with market data")
    parser.add_argument("--output", default="comprehensive_market_analysis.csv", help="Output CSV filename")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help=f"Maximum parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--sequential", action="store_true", help="Process markets sequentially (no parallelism)")
    parser.add_argument("--trades-dir", default=TRADES_DIR, help=f"Directory containing trade parquet files (default: {TRADES_DIR})")
    args = parser.parse_args()
    
    # Update trades directory if specified
    TRADES_DIR = args.trades_dir
    logger.info(f"Using trades directory: {TRADES_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load input CSV
    logger.info(f"Loading market data from {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} markets from CSV")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return
    
    # Define columns to include from original data
    include_columns = [
        'id', 'question', 'slug', 'startDate', 'endDate', 'description', 
        'outcomes', 'outcomePrices', 'volumeNum', 'volumeClob', 'active', 
        'groupItemTitle', 'clobTokenIds', 'event_id', 'event_ticker', 
        'event_slug', 'event_title', 'event_description', 'event_volume', 
        'event_negRisk', 'event_negRiskMarketID', 'event_commentCount', 
        'event_countryName', 'event_electionType'
    ]
    
    # Check column availability
    missing_columns = [col for col in include_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns in input CSV: {missing_columns}")
        include_columns = [col for col in include_columns if col in df.columns]
    
    # Process markets
    logger.info(f"Processing {len(df)} markets")
    results = []
    
    if args.sequential:
        # Process sequentially
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing markets"):
            result = process_market(row, include_columns)
            if result:
                results.append(result)
    else:
        # Process in parallel
        workers = min(args.max_workers, len(df))
        logger.info(f"Using {workers} parallel workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_market, row, include_columns) for _, row in df.iterrows()]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing markets"):
                result = future.result()
                if result:
                    results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, args.output)
    logger.info(f"Saving {len(results_df)} processed markets to {output_path}")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    logger.info(f"Processing complete: {len(results_df)}/{len(df)} markets successfully processed")
    logger.info(f"Results saved to {output_path}")
    
    # Print metrics statistics
    metrics_cols = [col for col in results_df.columns if col not in include_columns]
    if metrics_cols:
        logger.info("\nMetrics Statistics:")
        for col in metrics_cols:
            if col in results_df.columns:
                non_null = results_df[col].count()
                if pd.api.types.is_numeric_dtype(results_df[col]):
                    avg = results_df[col].mean()
                    logger.info(f"  {col}: {non_null}/{len(results_df)} markets ({non_null/len(results_df)*100:.1f}%), avg={avg:.4f}")
                else:
                    logger.info(f"  {col}: {non_null}/{len(results_df)} markets ({non_null/len(results_df)*100:.1f}%)")

if __name__ == "__main__":
    main()