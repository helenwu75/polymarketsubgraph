#!/usr/bin/env python3
"""
Election Market Metrics Calculator

This script calculates comprehensive metrics for Polymarket election markets
by processing trade data, orderbook data, and market information to create
features for machine learning analysis.
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
from typing import Dict, List, Any, Optional, Tuple, Set
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("election_metrics_calculation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 8  # Number of parallel workers
TRADES_DIR = "polymarket_raw_data/trades"
ORDERBOOKS_DIR = "polymarket_raw_data/orderbooks"
MATCHED_EVENTS_DIR = "polymarket_raw_data/matched_events"
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

def parse_outcomes(outcomes_str):
    """Parse outcomes from string representation in CSV."""
    try:
        if not outcomes_str or pd.isna(outcomes_str):
            return []
            
        # Handle different string formats
        if isinstance(outcomes_str, str):
            if outcomes_str.startswith('[') and outcomes_str.endswith(']'):
                try:
                    return json.loads(outcomes_str.replace("'", '"'))
                except:
                    return ast.literal_eval(outcomes_str)
            else:
                return outcomes_str.split(',')
        return []
    except Exception as e:
        logger.warning(f"Error parsing outcomes: {e}")
        return []

def parse_outcome_prices(prices_str):
    """Parse outcome prices from string representation in CSV."""
    try:
        if not prices_str or pd.isna(prices_str):
            return []
            
        # Handle different string formats
        if isinstance(prices_str, str):
            if prices_str.startswith('[') and prices_str.endswith(']'):
                try:
                    return json.loads(prices_str.replace("'", '"'))
                except:
                    return ast.literal_eval(prices_str)
            else:
                return prices_str.split(',')
        return []
    except Exception as e:
        logger.warning(f"Error parsing outcome prices: {e}")
        return []

def get_yes_token_id(row):
    """
    Identify the 'Yes' token ID for a binary market.
    For binary markets, we need to determine which token represents 'Yes'.
    
    Args:
        row: DataFrame row containing market data
        
    Returns:
        str: 'Yes' token ID, or None if cannot be determined
    """
    try:
        # Get token IDs
        token_ids = parse_token_ids(row.get('clobTokenIds', []))
        if not token_ids or len(token_ids) < 2:
            return None
            
        # Get outcomes
        outcomes = parse_outcomes(row.get('outcomes', []))
        if len(outcomes) >= 2:
            # If we have outcomes like ["Yes", "No"], find the index of "Yes"
            yes_index = -1
            for i, outcome in enumerate(outcomes):
                if outcome.lower() == "yes":
                    yes_index = i
                    break
                    
            if yes_index >= 0 and yes_index < len(token_ids):
                return token_ids[yes_index]
        
        # Fallback: Just use the first token, which is typically "Yes" in binary markets
        return token_ids[0]
        
    except Exception as e:
        logger.warning(f"Error determining 'Yes' token for market {row.get('id', 'unknown')}: {e}")
        return None

def determine_correct_outcome(row):
    """
    Determine the correct outcome for a market based on outcomePrices.
    For binary markets: outcomePrices = ["1", "0"] means "Yes" was correct,
    outcomePrices = ["0", "1"] means "No" was correct.
    
    Args:
        row: DataFrame row containing market data
        
    Returns:
        str: "Yes", "No", or None if cannot be determined
    """
    try:
        outcome_prices = parse_outcome_prices(row.get('outcomePrices', []))
        
        if len(outcome_prices) >= 2:
            if outcome_prices[0] == "1" and outcome_prices[1] == "0":
                return "Yes"
            elif outcome_prices[0] == "0" and outcome_prices[1] == "1":
                return "No"
        
        # If we can't determine from outcomePrices, check if there's a direct field
        if 'correct_outcome' in row:
            return row['correct_outcome']
            
        return None
            
    except Exception as e:
        logger.warning(f"Error determining correct outcome for market {row.get('id', 'unknown')}: {e}")
        return None

def load_token_trades(token_id):
    """
    Load trade data for a specific token from parquet file.
    
    Args:
        token_id: Token ID to load trades for
        
    Returns:
        DataFrame: Trades data, or empty DataFrame if not found
    """
    try:
        file_path = os.path.join(TRADES_DIR, f"{token_id}.parquet")
        
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns:
                if df['timestamp'].dtype == 'object' or np.issubdtype(df['timestamp'].dtype, np.integer):
                    # Ensure timestamp is numeric before conversion
                    df['timestamp_num'] = pd.to_numeric(df['timestamp'], errors='coerce')
                    df['datetime'] = pd.to_datetime(df['timestamp_num'], unit='s')
                elif pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['datetime'] = df['timestamp']
                
                # Add a datetime64 column for consistent comparisons
                df['datetime64'] = df['datetime'].astype('datetime64[ns]')
            
            # Ensure price is numeric
            if 'price' in df.columns:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                
            # Ensure size is numeric
            if 'size' in df.columns:
                df['size'] = pd.to_numeric(df['size'], errors='coerce')
                
            return df
        else:
            logger.warning(f"No trade data file found for token {token_id}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading trade data for token {token_id}: {e}")
        return pd.DataFrame()

def load_orderbook_data(token_id):
    """
    Load orderbook data for a specific token from JSON file.
    
    Args:
        token_id: Token ID to load orderbook data for
        
    Returns:
        dict: Orderbook data, or empty dict if not found
    """
    try:
        file_path = os.path.join(ORDERBOOKS_DIR, f"{token_id}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"No orderbook data file found for token {token_id}")
            return {}
            
    except Exception as e:
        logger.error(f"Error loading orderbook data for token {token_id}: {e}")
        return {}

def calculate_price_metrics(trades_df, market_end_date):
    """
    Calculate all price-based metrics for a token.
    
    Args:
        trades_df: DataFrame containing trade data
        market_end_date: Datetime object for market end
        
    Returns:
        dict: Calculated price metrics
    """
    metrics = {}
    
    try:
        if trades_df.empty or 'datetime' not in trades_df.columns:
            return metrics
            
        # Sort trades by datetime
        trades_df = trades_df.sort_values('datetime')
        
        # Convert market_end_date to datetime if it's a string
        if isinstance(market_end_date, str):
            market_end_date = pd.to_datetime(market_end_date)
            
        # Convert market_end_date to numpy datetime64 for consistent comparisons
        market_end_date_np = np.datetime64(market_end_date)
        
        # Define time periods
        one_day_before = np.datetime64(market_end_date - timedelta(days=1))
        two_days_before = np.datetime64(market_end_date - timedelta(days=2))
        week_before = np.datetime64(market_end_date - timedelta(days=7))
        
        # Convert trades_df['datetime'] to numpy datetime64
        trades_df['datetime64'] = trades_df['datetime'].astype('datetime64[ns]')
        
        # Filter trades for different time periods
        trades_before_end = trades_df[trades_df['datetime64'] <= market_end_date_np]
        trades_before_one_day = trades_df[trades_df['datetime64'] <= one_day_before]
        trades_before_two_days = trades_df[trades_df['datetime64'] <= two_days_before]
        trades_final_48h = trades_df[(trades_df['datetime64'] > two_days_before) & 
                                     (trades_df['datetime64'] <= market_end_date_np)]
        trades_final_week = trades_df[(trades_df['datetime64'] > week_before) & 
                                      (trades_df['datetime64'] <= market_end_date_np)]
        
        # 1. Closing Price (last trade before 24h prior to end)
        if not trades_before_one_day.empty:
            metrics['closing_price'] = float(trades_before_one_day.iloc[-1]['price'])
        
        # 2. Price 2 Days Prior
        if not trades_before_two_days.empty:
            metrics['price_2days_prior'] = float(trades_before_two_days.iloc[-1]['price'])
        
        # 3. Pre-election VWAP (48h)
        if not trades_final_48h.empty and 'size' in trades_final_48h.columns:
            trades_final_48h['value'] = trades_final_48h['price'] * trades_final_48h['size']
            total_value = trades_final_48h['value'].sum()
            total_size = trades_final_48h['size'].sum()
            
            if total_size > 0:
                metrics['pre_election_vwap_48h'] = float(total_value / total_size)
        
        # 4. Price Volatility (final week)
        if not trades_final_week.empty:
            price_std = trades_final_week['price'].std()
            price_mean = trades_final_week['price'].mean()
            
            if price_mean > 0:
                metrics['price_volatility'] = float(price_std / price_mean)
        
        # 5. Price Range (final week)
        if not trades_final_week.empty:
            price_max = trades_final_week['price'].max()
            price_min = trades_final_week['price'].min()
            metrics['price_range'] = float(price_max - price_min)
        
        # 6. Final Week Momentum
        if not trades_final_week.empty and len(trades_final_week) >= 2:
            first_price = trades_final_week.iloc[0]['price']
            last_price = trades_final_week.iloc[-1]['price']
            metrics['final_week_momentum'] = float(last_price - first_price)
        
        # 7. Price Fluctuations (crossing 0.5 threshold)
        if not trades_final_week.empty and len(trades_final_week) >= 2:
            prices = trades_final_week['price'].values
            crossings = 0
            
            for i in range(1, len(prices)):
                if (prices[i-1] < 0.5 and prices[i] >= 0.5) or (prices[i-1] >= 0.5 and prices[i] < 0.5):
                    crossings += 1
                    
            metrics['price_fluctuations'] = crossings
        
        # 8. Last Trade Price (absolute last price before market end)
        if not trades_before_end.empty:
            metrics['last_trade_price'] = float(trades_before_end.iloc[-1]['price'])
            
    except Exception as e:
        logger.warning(f"Error calculating price metrics: {e}")
    
    return metrics

def calculate_trading_metrics(trades_df, orderbook_data, market_start_date, market_end_date):
    """
    Calculate all trading activity metrics for a token.
    
    Args:
        trades_df: DataFrame containing trade data
        orderbook_data: Dict containing orderbook data
        market_start_date: Datetime object for market start
        market_end_date: Datetime object for market end
        
    Returns:
        dict: Calculated trading metrics
    """
    metrics = {}
    
    try:
        # Convert market dates to datetime if they're strings
        if isinstance(market_start_date, str):
            market_start_date = pd.to_datetime(market_start_date)
            
        if isinstance(market_end_date, str):
            market_end_date = pd.to_datetime(market_end_date)
            
        # Calculate market duration in days
        market_duration = max(1, (market_end_date - market_start_date).days)
        metrics['market_duration_days'] = market_duration
        
        # 1. Trading Frequency (trades per day)
        if not trades_df.empty:
            metrics['trading_frequency'] = len(trades_df) / market_duration
        
        # 2. Buy/Sell Ratio from orderbook data
        if orderbook_data:
            buys_quantity = int(orderbook_data.get('buysQuantity', 0))
            sells_quantity = int(orderbook_data.get('sellsQuantity', 0))
            
            if sells_quantity > 0:
                metrics['buy_sell_ratio'] = buys_quantity / sells_quantity
            elif buys_quantity > 0:
                metrics['buy_sell_ratio'] = float('inf')  # All buys, no sells
            else:
                metrics['buy_sell_ratio'] = 0
        
        # 3. Trading Continuity (percentage of days with trades)
        if not trades_df.empty and 'datetime' in trades_df.columns:
            # Get unique trading days
            trades_df['trade_date'] = trades_df['datetime'].dt.date
            trading_days = trades_df['trade_date'].nunique()
            
            metrics['trading_continuity'] = trading_days / market_duration
        
        # 4. Late Stage Participation (trades in final week / total trades)
        if not trades_df.empty and 'datetime64' in trades_df.columns:
            week_before_np = np.datetime64(market_end_date - timedelta(days=7))
            final_week_trades = trades_df[trades_df['datetime64'] > week_before_np]
            
            if len(trades_df) > 0:
                metrics['late_stage_participation'] = len(final_week_trades) / len(trades_df)
        
        # 5. Volume Acceleration (final week trading frequency vs. overall)
        if not trades_df.empty and 'datetime64' in trades_df.columns:
            week_before_np = np.datetime64(market_end_date - timedelta(days=7))
            final_week_trades = trades_df[trades_df['datetime64'] > week_before_np]
            
            overall_freq = len(trades_df) / market_duration
            
            if len(final_week_trades) > 0:
                final_week_freq = len(final_week_trades) / min(7, market_duration)
                
                if overall_freq > 0:
                    metrics['volume_acceleration'] = final_week_freq / overall_freq
            
    except Exception as e:
        logger.warning(f"Error calculating trading metrics: {e}")
    
    return metrics

def calculate_trader_metrics(trades_df, market_end_date):
    """
    Calculate all trader-based metrics for a token.
    
    Args:
        trades_df: DataFrame containing trade data
        market_end_date: Datetime object for market end
        
    Returns:
        dict: Calculated trader metrics
    """
    metrics = {}
    
    try:
        if trades_df.empty:
            return metrics
            
        # Ensure we have the required columns
        required_columns = ['maker_id', 'taker_id', 'side', 'datetime64']
        missing_columns = [col for col in required_columns if col not in trades_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns for trader metrics: {missing_columns}")
            return metrics
            
        # Convert market_end_date to datetime if it's a string
        if isinstance(market_end_date, str):
            market_end_date = pd.to_datetime(market_end_date)
            
        # Define time periods
        week_before_np = np.datetime64(market_end_date - timedelta(days=7))
        
        # Get all unique traders
        all_makers = set(trades_df['maker_id'].dropna().unique())
        all_takers = set(trades_df['taker_id'].dropna().unique())
        all_traders = all_makers.union(all_takers)
        
        # 1. Unique Traders Count
        metrics['unique_traders_count'] = len(all_traders)
        
        # 2. Trader to Trade Ratio
        if metrics['unique_traders_count'] > 0:
            metrics['trader_to_trade_ratio'] = len(trades_df) / metrics['unique_traders_count']
        
        # 3. Two-Way Traders Ratio
        buyers = set()
        sellers = set()
        
        # Find all buyers
        buy_trades = trades_df[trades_df['side'] == 'Buy']
        for _, trade in buy_trades.iterrows():
            if pd.notna(trade['taker_id']):
                buyers.add(trade['taker_id'])
        
        # Find all sellers
        sell_trades = trades_df[trades_df['side'] == 'Sell']
        for _, trade in sell_trades.iterrows():
            if pd.notna(trade['taker_id']):
                sellers.add(trade['taker_id'])
        
        # Calculate two-way traders
        two_way_traders = buyers.intersection(sellers)
        
        if len(all_traders) > 0:
            metrics['two_way_traders_ratio'] = len(two_way_traders) / len(all_traders)
        
        # 4. Trader Concentration
        trader_activity = {}
        
        # Count trades per trader
        for _, trade in trades_df.iterrows():
            if pd.notna(trade['maker_id']):
                trader_id = trade['maker_id']
                trader_activity[trader_id] = trader_activity.get(trader_id, 0) + 1
            
            if pd.notna(trade['taker_id']):
                trader_id = trade['taker_id']
                trader_activity[trader_id] = trader_activity.get(trader_id, 0) + 1
        
        if trader_activity:
            # Sort traders by activity
            sorted_traders = sorted(trader_activity.items(), key=lambda x: x[1], reverse=True)
            
            # Top 10% traders
            top_count = max(1, int(len(sorted_traders) * 0.1))
            top_traders = sorted_traders[:top_count]
            
            # Count trades by top traders
            top_trader_trades = sum(count for _, count in top_traders)
            
            # Calculate concentration
            metrics['trader_concentration'] = top_trader_trades / len(trades_df)
        
        # 5. New Trader Influx
        if 'datetime64' in trades_df.columns:
            final_week_trades = trades_df[trades_df['datetime64'] > week_before_np]
            
            # Get traders who appeared in the final week
            final_week_makers = set(final_week_trades['maker_id'].dropna().unique())
            final_week_takers = set(final_week_trades['taker_id'].dropna().unique())
            final_week_traders = final_week_makers.union(final_week_takers)
            
            # Identify traders who only appeared in the final week
            # First, get all traders before the final week
            earlier_trades = trades_df[trades_df['datetime64'] <= week_before_np]
            earlier_makers = set(earlier_trades['maker_id'].dropna().unique())
            earlier_takers = set(earlier_trades['taker_id'].dropna().unique())
            earlier_traders = earlier_makers.union(earlier_takers)
            
            # Traders who only appeared in the final week
            new_traders = final_week_traders - earlier_traders
            
            if len(all_traders) > 0:
                metrics['new_trader_influx'] = len(new_traders) / len(all_traders)
                
    except Exception as e:
        logger.warning(f"Error calculating trader metrics: {e}")
    
    return metrics

def calculate_prediction_accuracy(price_metrics, correct_outcome):
    """
    Calculate prediction accuracy metrics based on the closing price and actual outcome.
    
    Args:
        price_metrics: Dict containing price metrics
        correct_outcome: The correct outcome ("Yes" or "No")
        
    Returns:
        dict: Prediction accuracy metrics
    """
    metrics = {}
    
    try:
        # Skip if we don't have the required data
        if not price_metrics or not correct_outcome or 'closing_price' not in price_metrics:
            return metrics
            
        closing_price = price_metrics['closing_price']
        
        # Determine if the prediction was correct
        # For "Yes" outcome: higher price = more accurate
        # For "No" outcome: lower price = more accurate
        if correct_outcome == "Yes":
            # Price above 0.5 indicates "Yes" prediction
            prediction_correct = closing_price > 0.5
            prediction_error = abs(1 - closing_price)
        else:  # "No" outcome
            # Price below 0.5 indicates "No" prediction
            prediction_correct = closing_price < 0.5
            prediction_error = abs(0 - closing_price)
        
        metrics['prediction_correct'] = int(prediction_correct)
        metrics['prediction_error'] = prediction_error
        metrics['prediction_confidence'] = abs(closing_price - 0.5) * 2  # Scale to [0, 1]
        
    except Exception as e:
        logger.warning(f"Error calculating prediction accuracy: {e}")
    
    return metrics

def process_market(row, include_columns):
    """
    Process a single market to calculate all metrics.
    
    Args:
        row: DataFrame row containing market data
        include_columns: Columns to include in the output
        
    Returns:
        dict: All calculated metrics for this market
    """
    try:
        # Create output dictionary with original fields
        result = {col: row[col] for col in include_columns if col in row}
        
        # Parse dates
        market_start_date = None
        market_end_date = None
        
        if 'startDate' in row and not pd.isna(row['startDate']):
            try:
                start_date_str = row['startDate'].replace('Z', '+00:00') if isinstance(row['startDate'], str) else row['startDate']
                market_start_date = pd.to_datetime(start_date_str)
                result['market_start_date'] = market_start_date
            except Exception as e:
                logger.warning(f"Could not parse startDate: {row['startDate']} - {e}")
        
        if 'endDate' in row and not pd.isna(row['endDate']):
            try:
                end_date_str = row['endDate'].replace('Z', '+00:00') if isinstance(row['endDate'], str) else row['endDate']
                market_end_date = pd.to_datetime(end_date_str)
                result['market_end_date'] = market_end_date
            except Exception as e:
                logger.warning(f"Could not parse endDate: {row['endDate']} - {e}")
        
        # Determine the correct outcome for this market
        correct_outcome = determine_correct_outcome(row)
        if correct_outcome:
            result['correct_outcome'] = correct_outcome
        
        # Identify the "Yes" token for this market
        yes_token_id = get_yes_token_id(row)
        if yes_token_id:
            result['yes_token_id'] = yes_token_id
            
            # Load trade data for the "Yes" token
            trades_df = load_token_trades(yes_token_id)
            
            # Load orderbook data for the "Yes" token
            orderbook_data = load_orderbook_data(yes_token_id)
            
            if not trades_df.empty:
                # Calculate price-based metrics
                if market_end_date:
                    price_metrics = calculate_price_metrics(trades_df, market_end_date)
                    result.update(price_metrics)
                    
                    # Calculate prediction accuracy
                    if correct_outcome:
                        accuracy_metrics = calculate_prediction_accuracy(price_metrics, correct_outcome)
                        result.update(accuracy_metrics)
                
                # Calculate trading activity metrics
                if market_start_date and market_end_date:
                    trading_metrics = calculate_trading_metrics(trades_df, orderbook_data, market_start_date, market_end_date)
                    result.update(trading_metrics)
                
                # Calculate trader-based metrics
                if market_end_date:
                    trader_metrics = calculate_trader_metrics(trades_df, market_end_date)
                    result.update(trader_metrics)
            else:
                logger.warning(f"No trade data found for market {row.get('id', 'unknown')}, yes_token_id={yes_token_id}")
        else:
            logger.warning(f"Could not identify 'Yes' token for market {row.get('id', 'unknown')}")
        
        return result
        
    except Exception as e:
        import traceback
        logger.error(f"Error processing market {row.get('id', 'unknown')}: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        
        # Return basic data without calculated metrics
        return {col: row[col] for col in include_columns if col in row}

def main():
    parser = argparse.ArgumentParser(description="Election Market Metrics Calculator")
    parser.add_argument("--input", required=True, help="Input CSV file with market data")
    parser.add_argument("--output", default="election_market_metrics.csv", help="Output CSV filename")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help=f"Maximum parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--sequential", action="store_true", help="Process markets sequentially (no parallelism)")
    parser.add_argument("--max-markets", type=int, help="Maximum number of markets to process")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load input CSV
    logger.info(f"Loading market data from {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} markets from CSV")
        
        # Limit number of markets if specified
        if args.max_markets and len(df) > args.max_markets:
            logger.info(f"Limiting to {args.max_markets} markets")
            df = df.head(args.max_markets)
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return
    
    # Define columns to include from original data
    include_columns = [
        'id', 'question', 'groupItemTitle','slug', 'startDate', 'endDate', 'description', 
        'outcomes', 'outcomePrices', 'volumeNum', 'volumeClob', 'active', 
        'clobTokenIds', 'event_id', 'event_ticker', 'event_slug', 
        'event_title', 'event_description', 'event_volume', 
        'event_countryName', 'event_electionType', 'event_commentCount','restricted','enableOrderBook'
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

    # Optionally, generate a summary of prediction accuracy
    if 'prediction_correct' in results_df.columns:
        correct_count = results_df['prediction_correct'].sum()
        total_count = results_df['prediction_correct'].count()
        if total_count > 0:
            accuracy = correct_count / total_count * 100
            logger.info(f"\nPrediction Accuracy: {correct_count}/{total_count} markets ({accuracy:.1f}%)")
            
            # Breakdown by election type if available
            if 'event_electionType' in results_df.columns:
                logger.info("\nAccuracy by Election Type:")
                for election_type, group in results_df.groupby('event_electionType'):
                    if pd.notna(election_type) and len(group) > 0:
                        type_correct = group['prediction_correct'].sum()
                        type_total = group['prediction_correct'].count()
                        type_accuracy = type_correct / type_total * 100 if type_total > 0 else 0
                        logger.info(f"  {election_type}: {type_correct}/{type_total} ({type_accuracy:.1f}%)")
            
            # Breakdown by country if available
            if 'event_countryName' in results_df.columns:
                logger.info("\nAccuracy by Country:")
                country_stats = []
                for country, group in results_df.groupby('event_countryName'):
                    if pd.notna(country) and len(group) > 0:
                        country_correct = group['prediction_correct'].sum()
                        country_total = group['prediction_correct'].count()
                        country_accuracy = country_correct / country_total * 100 if country_total > 0 else 0
                        country_stats.append((country, country_correct, country_total, country_accuracy))
                
                # Sort by count and show top countries
                for country, correct, total, accuracy in sorted(country_stats, key=lambda x: x[2], reverse=True)[:10]:
                    logger.info(f"  {country}: {correct}/{total} ({accuracy:.1f}%)")

    # Generate a correlation matrix for key metrics
    key_metrics = [
        'closing_price', 'price_volatility', 'buy_sell_ratio', 'trader_concentration',
        'trading_frequency', 'late_stage_participation', 'two_way_traders_ratio',
        'prediction_correct', 'prediction_error'
    ]
    
    # Filter to metrics actually present in the results
    available_metrics = [col for col in key_metrics if col in results_df.columns]
    
    if len(available_metrics) >= 2:
        try:
            # Calculate correlation matrix
            corr_matrix = results_df[available_metrics].corr()
            
            # Save correlation matrix to CSV
            corr_file = os.path.join(OUTPUT_DIR, "metric_correlations.csv")
            corr_matrix.to_csv(corr_file)
            logger.info(f"\nMetric correlations saved to {corr_file}")
            
            # Log correlations with prediction_correct if available
            if 'prediction_correct' in available_metrics:
                logger.info("\nCorrelations with prediction correctness:")
                for col in available_metrics:
                    if col != 'prediction_correct':
                        corr = corr_matrix.loc['prediction_correct', col]
                        logger.info(f"  {col}: {corr:.4f}")
        except Exception as e:
            logger.warning(f"Error generating correlation matrix: {e}")

if __name__ == "__main__":
    main()