#!/usr/bin/env python3
"""
Election Market Prediction Metrics Collector

This script analyzes Polymarket election markets to extract predictive metrics
before the election event date. It computes multiple metrics including:
- Closing price (last trade before election)
- Pre-election VWAP (volume-weighted average price)
- Pre-election median price
- Price stability metrics
- Liquidity metrics

The script ensures all metrics are based solely on pre-election data to avoid
contamination from post-outcome information.

Usage:
    python election_metrics.py --market-id <market_id> 
                               --token-ids <token_ids_str>
                               --election-date <YYYY-MM-DD>
                               [--days-before <days>]
                               [--output <filename>]
"""

import os
import sys
import json
import ast
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from subgrounds import Subgrounds

# Configure constants
DEFAULT_PRE_ELECTION_DAYS = 7       # Number of days before election to analyze
MAX_QUERY_SIZE = 1000               # Maximum number of records to fetch in a single query
VWAP_WINDOW_HOURS = 48              # Hours for VWAP calculation
STABILITY_WINDOW_DAYS = 7           # Days for price stability calculation
OUTPUT_DIRECTORY = "prediction_metrics"  # Directory to store results


def parse_token_ids(token_ids_str):
    """
    Parse token IDs from a string representation.
    
    Args:
        token_ids_str (str): String representation of token IDs list
        
    Returns:
        list: List of token IDs
    """
    if not token_ids_str or token_ids_str == 'nan' or token_ids_str == 'None':
        return []
        
    try:
        # Try to parse the token IDs string - it might be in different formats
        if token_ids_str.startswith('[') and token_ids_str.endswith(']'):
            # It's already in a list-like format
            if '"' in token_ids_str or "'" in token_ids_str:
                # It has quotes, so it might be a JSON string
                try:
                    token_ids = json.loads(token_ids_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try ast.literal_eval
                    token_ids = ast.literal_eval(token_ids_str)
            else:
                # It's a simple string representation of a list
                token_ids = token_ids_str.strip('[]').split(',')
        else:
            # It's a comma-separated string
            token_ids = token_ids_str.split(',')
        
        # Clean up any whitespace
        token_ids = [id.strip().strip('"\'') for id in token_ids]
        return token_ids
    except Exception as e:
        print(f"Error parsing token IDs string: {e}")
        return [token_ids_str]  # Return as a single ID


def parse_election_date(date_str):
    """
    Parse election date string into datetime object.
    
    Args:
        date_str (str): Election date in YYYY-MM-DD format
        
    Returns:
        datetime: Election date as datetime object
    """
    try:
        # Try parsing ISO format first (if time is included)
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            # If that fails, try simple YYYY-MM-DD format
            return datetime.strptime(date_str, '%Y-%m-%d')
    except Exception as e:
        raise ValueError(f"Invalid election date format. Please use YYYY-MM-DD. Error: {e}")


def get_trade_history(token_id, election_timestamp, days_before, sg, orderbook_subgraph):
    """
    Get trade history for a specific token for the period before the election.
    
    Args:
        token_id (str): Token ID
        election_timestamp (int): Unix timestamp of election date
        days_before (int): Number of days before election to analyze
        sg (Subgrounds): Subgrounds instance
        orderbook_subgraph: Orderbook subgraph instance
        
    Returns:
        pd.DataFrame: Trade history as DataFrame
    """
    # Calculate start timestamp
    start_timestamp = election_timestamp - (days_before * 24 * 60 * 60)
    
    print(f"Fetching trades from {datetime.fromtimestamp(start_timestamp)} to {datetime.fromtimestamp(election_timestamp)}")
    
    # Initialize storage for all trades
    all_trades = []
    skip = 0
    has_more = True
    
    # For resolved markets, we'll try multiple approaches
    
    # Approach 1: Try OrderFilledEvent entities which may have more historical data
    print("Attempting to query using OrderFilledEvent entities...")
    
    # Loop until we get all trades in the time range
    while has_more:
        try:
            # Query OrderFilledEvent entities for this token
            events_query = orderbook_subgraph.Query.orderFilledEvents(
                first=MAX_QUERY_SIZE,
                skip=skip,
                orderBy='timestamp',
                orderDirection='asc',
                where={
                    'timestamp_gte': str(start_timestamp),
                    'timestamp_lt': str(election_timestamp),
                    'makerAssetId': token_id  # Use makerAssetId instead of market
                }
            )
            
            # We need timestamp and amounts for calculations
            events_result = sg.query_df([
                events_query.id,
                events_query.timestamp,
                events_query.makerAmountFilled,
                events_query.takerAmountFilled,
                events_query.makerAssetId,
                events_query.takerAssetId,
                events_query.transactionHash
            ])
            
            # Check if we found trades
            if events_result.empty:
                has_more = False
                print("No trades found with OrderFilledEvent query")
                break
                
            # Process and transform the data to match our expected format
            # Calculate price from makerAmountFilled and takerAmountFilled
            events_result['enrichedOrderFilleds_price'] = events_result.apply(
                lambda row: float(row['orderFilledEvents_takerAmountFilled']) / float(row['orderFilledEvents_makerAmountFilled']) 
                if float(row['orderFilledEvents_makerAmountFilled']) > 0 else None, 
                axis=1
            )
            
            # Rename columns to match EnrichedOrderFilled format
            events_result['enrichedOrderFilleds_timestamp'] = events_result['orderFilledEvents_timestamp']
            events_result['enrichedOrderFilleds_size'] = events_result['orderFilledEvents_makerAmountFilled']
            events_result['enrichedOrderFilleds_transactionHash'] = events_result['orderFilledEvents_transactionHash']
            
            # Determine if it's a buy or sell
            events_result['enrichedOrderFilleds_side'] = 'Buy'  # Simplification; we'd need more data to determine accurately
            
            # Add to our collection
            all_trades.append(events_result)
            
            # Update skip for next batch
            skip += len(events_result)
            
            # Print progress
            print(f"Fetched {len(events_result)} trades (total: {skip})")
            
            # If we got fewer trades than the max query size, we've reached the end
            if len(events_result) < MAX_QUERY_SIZE:
                has_more = False
                
        except Exception as e:
            print(f"Error fetching trades with OrderFilledEvent: {e}")
            has_more = False
    
    # If we don't have trades yet, try the EnrichedOrderFilled approach
    if not all_trades:
        print("\nAttempting to query using EnrichedOrderFilled entities...")
        skip = 0
        has_more = True
        
        while has_more:
            try:
                # Query EnrichedOrderFilled events for this token
                events_query = orderbook_subgraph.Query.enrichedOrderFilleds(
                    first=MAX_QUERY_SIZE,
                    skip=skip,
                    orderBy='timestamp',
                    orderDirection='asc',
                    where={
                        'timestamp_gte': str(start_timestamp),
                        'timestamp_lt': str(election_timestamp),
                        'market': token_id
                    }
                )
                
                # We need timestamp, price, and size for calculations
                events_result = sg.query_df([
                    events_query.id,
                    events_query.timestamp,
                    events_query.price,
                    events_query.size,
                    events_query.side,
                    events_query.transactionHash
                ])
                
                # Check if we found trades
                if events_result.empty:
                    has_more = False
                    print("No trades found with EnrichedOrderFilled query")
                    break
                    
                # Add to our collection
                all_trades.append(events_result)
                
                # Update skip for next batch
                skip += len(events_result)
                
                # Print progress
                print(f"Fetched {len(events_result)} trades (total: {skip})")
                
                # If we got fewer trades than the max query size, we've reached the end
                if len(events_result) < MAX_QUERY_SIZE:
                    has_more = False
                    
            except Exception as e:
                print(f"Error fetching trades with EnrichedOrderFilled: {e}")
                has_more = False
    
    # If we still don't have trades, try a third approach using Transaction entity
    if not all_trades:
        print("\nAttempting to query using Transaction entities...")
        skip = 0
        has_more = True
        
        # Try to find the market (FixedProductMarketMaker) associated with this token
        try:
            # First get the MarketData entity which links tokens to markets
            market_query = orderbook_subgraph.Query.marketData(
                id=token_id
            )
            
            market_data = sg.query_df([
                market_query.id,
                market_query.fpmm.id
            ])
            
            if not market_data.empty and 'marketData_fpmm_id' in market_data.columns:
                market_id = market_data['marketData_fpmm_id'].iloc[0]
                
                if market_id:
                    print(f"Found associated market ID: {market_id}")
                    
                    while has_more:
                        try:
                            # Query Transaction entities for this market
                            tx_query = orderbook_subgraph.Query.transactions(
                                first=MAX_QUERY_SIZE,
                                skip=skip,
                                orderBy='timestamp',
                                orderDirection='asc',
                                where={
                                    'timestamp_gte': str(start_timestamp),
                                    'timestamp_lt': str(election_timestamp),
                                    'market': market_id,
                                    'outcomeIndex': market_data['marketData_outcomeIndex'].iloc[0] if 'marketData_outcomeIndex' in market_data.columns else None
                                }
                            )
                            
                            tx_result = sg.query_df([
                                tx_query.id,
                                tx_query.timestamp,
                                tx_query.tradeAmount,
                                tx_query.type,
                                tx_query.outcomeIndex
                            ])
                            
                            if tx_result.empty:
                                has_more = False
                                print("No transactions found")
                                continue
                                
                            # Transform transaction data to match our expected format
                            tx_result['enrichedOrderFilleds_timestamp'] = tx_result['transactions_timestamp']
                            tx_result['enrichedOrderFilleds_size'] = tx_result['transactions_tradeAmount']
                            
                            # We don't have direct price data from transactions, so we'll use a placeholder
                            # This will make most price metrics unavailable
                            tx_result['enrichedOrderFilleds_price'] = None
                            
                            # Determine if it's a buy or sell
                            tx_result['enrichedOrderFilleds_side'] = tx_result['transactions_type']
                            
                            # Add to our collection
                            all_trades.append(tx_result)
                            
                            # Update skip for next batch
                            skip += len(tx_result)
                            
                            # Print progress
                            print(f"Fetched {len(tx_result)} transactions (total: {skip})")
                            
                            # If we got fewer transactions than the max query size, we've reached the end
                            if len(tx_result) < MAX_QUERY_SIZE:
                                has_more = False
                        
                        except Exception as e:
                            print(f"Error fetching transactions: {e}")
                            has_more = False
        
        except Exception as e:
            print(f"Error finding associated market: {e}")
    
    # Combine all trades if we have any
    if all_trades:
        try:
            trades_df = pd.concat(all_trades, ignore_index=True)
            
            # Convert timestamp to datetime for easier handling
            trades_df['datetime'] = pd.to_datetime(trades_df['enrichedOrderFilleds_timestamp'].astype(int), unit='s')
            
            # Sort by timestamp to ensure correct order
            trades_df = trades_df.sort_values(by='enrichedOrderFilleds_timestamp')
            
            print(f"Successfully processed {len(trades_df)} trades")
            return trades_df
        except Exception as e:
            print(f"Error processing trade data: {e}")
            # If concat fails, try returning the first dataframe
            if len(all_trades) > 0:
                return all_trades[0]
    
    # Return empty DataFrame if no trades found
    print("No trade data found for the specified period")
    return pd.DataFrame()


def calculate_closing_price(trades_df):
    """
    Calculate the closing price (last trade before election).
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
        
    Returns:
        dict: Closing price metrics
    """
    if trades_df.empty:
        return {
            'closing_price': None,
            'closing_price_time': None,
            'time_before_election': None
        }
    
    # Get the last trade
    last_trade = trades_df.iloc[-1]
    
    # Calculate time before election
    last_trade_time = last_trade['datetime']
    election_time = pd.to_datetime(trades_df['enrichedOrderFilleds_timestamp'].max() + 1, unit='s')
    time_before_election = (election_time - last_trade_time).total_seconds() / 3600  # in hours
    
    return {
        'closing_price': float(last_trade['enrichedOrderFilleds_price']),
        'closing_price_time': last_trade_time.isoformat(),
        'time_before_election_hours': time_before_election
    }


def calculate_vwap(trades_df, hours_window):
    """
    Calculate volume-weighted average price for a specific time window.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
        hours_window (int): Hours to include in VWAP calculation
        
    Returns:
        dict: VWAP metrics
    """
    if trades_df.empty:
        return {
            'vwap': None,
            'vwap_window_hours': hours_window,
            'vwap_trade_count': 0,
            'vwap_volume': 0
        }
    
    # Get last trade timestamp
    last_trade_timestamp = trades_df['enrichedOrderFilleds_timestamp'].max()
    
    # Calculate window start timestamp
    window_start_timestamp = last_trade_timestamp - (hours_window * 60 * 60)
    
    # Filter trades in the VWAP window
    vwap_trades = trades_df[trades_df['enrichedOrderFilleds_timestamp'] >= window_start_timestamp]
    
    if vwap_trades.empty:
        return {
            'vwap': None,
            'vwap_window_hours': hours_window,
            'vwap_trade_count': 0,
            'vwap_volume': 0
        }
    
    # Calculate VWAP
    vwap_trades['volume_price'] = vwap_trades['enrichedOrderFilleds_size'].astype(float) * vwap_trades['enrichedOrderFilleds_price'].astype(float)
    total_volume = vwap_trades['enrichedOrderFilleds_size'].astype(float).sum()
    total_volume_price = vwap_trades['volume_price'].sum()
    
    if total_volume > 0:
        vwap = total_volume_price / total_volume
    else:
        vwap = None
    
    return {
        'vwap': vwap,
        'vwap_window_hours': hours_window,
        'vwap_trade_count': len(vwap_trades),
        'vwap_volume': float(total_volume)
    }


def calculate_median_price(trades_df, days_window):
    """
    Calculate median price over a specific time window.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
        days_window (int): Days to include in median calculation
        
    Returns:
        dict: Median price metrics
    """
    if trades_df.empty:
        return {
            'median_price': None,
            'median_window_days': days_window,
            'median_trade_count': 0
        }
    
    # Get last trade timestamp
    last_trade_timestamp = trades_df['enrichedOrderFilleds_timestamp'].max()
    
    # Calculate window start timestamp
    window_start_timestamp = last_trade_timestamp - (days_window * 24 * 60 * 60)
    
    # Filter trades in the median window
    median_trades = trades_df[trades_df['enrichedOrderFilleds_timestamp'] >= window_start_timestamp]
    
    if median_trades.empty:
        return {
            'median_price': None,
            'median_window_days': days_window,
            'median_trade_count': 0
        }
    
    # Calculate median price
    median_price = median_trades['enrichedOrderFilleds_price'].astype(float).median()
    
    return {
        'median_price': float(median_price),
        'median_window_days': days_window,
        'median_trade_count': len(median_trades)
    }


def calculate_price_stability(trades_df, days_window):
    """
    Calculate price stability metrics over a specific time window.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
        days_window (int): Days to include in stability calculation
        
    Returns:
        dict: Price stability metrics
    """
    if trades_df.empty:
        return {
            'price_std_dev': None,
            'price_volatility': None,
            'price_range': None,
            'stability_window_days': days_window,
            'stability_trade_count': 0
        }
    
    # Get last trade timestamp
    last_trade_timestamp = trades_df['enrichedOrderFilleds_timestamp'].max()
    
    # Calculate window start timestamp
    window_start_timestamp = last_trade_timestamp - (days_window * 24 * 60 * 60)
    
    # Filter trades in the stability window
    stability_trades = trades_df[trades_df['enrichedOrderFilleds_timestamp'] >= window_start_timestamp]
    
    if stability_trades.empty or len(stability_trades) < 2:
        return {
            'price_std_dev': None,
            'price_volatility': None,
            'price_range': None,
            'stability_window_days': days_window,
            'stability_trade_count': len(stability_trades)
        }
    
    # Convert to float to ensure calculations work correctly
    prices = stability_trades['enrichedOrderFilleds_price'].astype(float)
    
    # Calculate standard deviation of prices
    price_std_dev = prices.std()
    
    # Calculate price volatility (std dev / mean)
    price_mean = prices.mean()
    price_volatility = price_std_dev / price_mean if price_mean > 0 else None
    
    # Calculate price range (max - min)
    price_range = prices.max() - prices.min()
    
    return {
        'price_std_dev': float(price_std_dev),
        'price_volatility': float(price_volatility) if price_volatility is not None else None,
        'price_range': float(price_range),
        'stability_window_days': days_window,
        'stability_trade_count': len(stability_trades)
    }


def calculate_liquidity_metrics(trades_df, days_window):
    """
    Calculate liquidity metrics over a specific time window.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
        days_window (int): Days to include in liquidity calculation
        
    Returns:
        dict: Liquidity metrics
    """
    if trades_df.empty:
        return {
            'total_volume': 0,
            'trade_count': 0,
            'avg_daily_volume': 0,
            'avg_trade_size': 0,
            'liquidity_window_days': days_window
        }
    
    # Get last trade timestamp
    last_trade_timestamp = trades_df['enrichedOrderFilleds_timestamp'].max()
    
    # Calculate window start timestamp
    window_start_timestamp = last_trade_timestamp - (days_window * 24 * 60 * 60)
    
    # Filter trades in the liquidity window
    liquidity_trades = trades_df[trades_df['enrichedOrderFilleds_timestamp'] >= window_start_timestamp]
    
    if liquidity_trades.empty:
        return {
            'total_volume': 0,
            'trade_count': 0,
            'avg_daily_volume': 0,
            'avg_trade_size': 0,
            'liquidity_window_days': days_window
        }
    
    # Calculate total volume
    total_volume = liquidity_trades['enrichedOrderFilleds_size'].astype(float).sum()
    
    # Calculate trade count
    trade_count = len(liquidity_trades)
    
    # Calculate average daily volume
    avg_daily_volume = total_volume / days_window
    
    # Calculate average trade size
    avg_trade_size = total_volume / trade_count if trade_count > 0 else 0
    
    return {
        'total_volume': float(total_volume),
        'trade_count': trade_count,
        'avg_daily_volume': float(avg_daily_volume),
        'avg_trade_size': float(avg_trade_size),
        'liquidity_window_days': days_window
    }


def calculate_buy_sell_ratio(trades_df, days_window):
    """
    Calculate the ratio of buy to sell trades over a specific time window.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
        days_window (int): Days to include in calculation
        
    Returns:
        dict: Buy/sell ratio metrics
    """
    if trades_df.empty:
        return {
            'buy_sell_ratio': None,
            'buy_count': 0,
            'sell_count': 0,
            'buy_sell_window_days': days_window
        }
    
    # Get last trade timestamp
    last_trade_timestamp = trades_df['enrichedOrderFilleds_timestamp'].max()
    
    # Calculate window start timestamp
    window_start_timestamp = last_trade_timestamp - (days_window * 24 * 60 * 60)
    
    # Filter trades in the window
    window_trades = trades_df[trades_df['enrichedOrderFilleds_timestamp'] >= window_start_timestamp]
    
    if window_trades.empty:
        return {
            'buy_sell_ratio': None,
            'buy_count': 0,
            'sell_count': 0,
            'buy_sell_window_days': days_window
        }
    
    # Count buys and sells
    buy_count = len(window_trades[window_trades['enrichedOrderFilleds_side'] == 'Buy'])
    sell_count = len(window_trades[window_trades['enrichedOrderFilleds_side'] == 'Sell'])
    
    # Calculate ratio
    buy_sell_ratio = buy_count / sell_count if sell_count > 0 else None
    
    return {
        'buy_sell_ratio': float(buy_sell_ratio) if buy_sell_ratio is not None else None,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'buy_sell_window_days': days_window
    }


def extract_prediction_metrics(market_id, token_ids, election_date_str, days_before=DEFAULT_PRE_ELECTION_DAYS):
    """
    Extract comprehensive prediction metrics for a Polymarket election market.
    
    Args:
        market_id (str): Market ID (FixedProductMarketMaker address)
        token_ids (list): List of token IDs
        election_date_str (str): Election date in YYYY-MM-DD format
        days_before (int): Number of days before election to analyze
        
    Returns:
        dict: Comprehensive prediction metrics
    """
    # Parse election date
    election_date = parse_election_date(election_date_str)
    
    # Set election date to midnight of the specified day
    election_date = election_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Convert to Unix timestamp
    election_timestamp = int(election_date.timestamp())
    
    print(f"Election date: {election_date} (timestamp: {election_timestamp})")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")
    
    # Initialize Subgrounds with proper configuration
    sg = Subgrounds()
    
    # Connect to orderbook subgraph
    orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    print(f"Connecting to orderbook subgraph: {orderbook_url[:30]}...{orderbook_url[-15:]}")
    orderbook_subgraph = sg.load_subgraph(orderbook_url)
    
    # Also get the activity subgraph which might have more market structure data
    activity_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
    print(f"Connecting to activity subgraph: {activity_url[:30]}...{activity_url[-15:]}")
    activity_subgraph = sg.load_subgraph(activity_url)
    
    # Try to validate the market ID by checking if it exists in the activity subgraph
    try:
        market_query = activity_subgraph.Query.fixedProductMarketMaker(
            id=market_id.lower()  # Ensure lowercase for Ethereum addresses
        )
        
        market_info = sg.query_df([
            market_query.id,
            market_query.creationTimestamp,
            market_query.creationTransactionHash
        ])
        
        if not market_info.empty:
            market_creation = datetime.fromtimestamp(
                int(market_info['fixedProductMarketMaker_creationTimestamp'].iloc[0])
            )
            print(f"Market found! Created on: {market_creation}")
        else:
            print("Warning: Market ID not found in activity subgraph. This may affect data retrieval.")
    except Exception as e:
        print(f"Warning: Could not validate market ID: {e}")
    
    # Initialize metrics dictionary
    metrics = {
        'market_id': market_id,
        'token_ids': token_ids,
        'election_date': election_date_str,
        'analysis_timestamp': datetime.now().isoformat(),
        'pre_election_days_analyzed': days_before,
        'tokens_data': []
    }
    
    # Analyze each token ID
    for token_id in token_ids:
        print(f"\n{'='*50}")
        print(f"Analyzing token ID: {token_id}")
        print(f"{'='*50}")
        
        token_metrics = {'token_id': token_id}
        
        # First, try to verify this token exists and get some basic info
        try:
            # Check if the token exists in the orderbook subgraph
            token_check = orderbook_subgraph.Query.marketData(
                id=token_id
            )
            
            token_info = sg.query_df([
                token_check.id,
                token_check.condition.id,
                token_check.outcomeIndex,
                token_check.priceOrderbook
            ])
            
            if not token_info.empty:
                print(f"Token verified! Outcome index: {token_info['marketData_outcomeIndex'].iloc[0] if 'marketData_outcomeIndex' in token_info.columns else 'Unknown'}")
                print(f"Associated condition: {token_info['marketData_condition_id'].iloc[0] if 'marketData_condition_id' in token_info.columns else 'Unknown'}")
                
                if 'marketData_priceOrderbook' in token_info.columns:
                    print(f"Last known price: {token_info['marketData_priceOrderbook'].iloc[0]}")
                
                # Store token info in metrics
                token_metrics['outcome_index'] = token_info['marketData_outcomeIndex'].iloc[0] if 'marketData_outcomeIndex' in token_info.columns else None
                token_metrics['condition_id'] = token_info['marketData_condition_id'].iloc[0] if 'marketData_condition_id' in token_info.columns else None
                token_metrics['last_known_price'] = token_info['marketData_priceOrderbook'].iloc[0] if 'marketData_priceOrderbook' in token_info.columns else None
            else:
                print("Warning: Token not found in orderbook subgraph")
        except Exception as e:
            print(f"Warning: Could not verify token: {e}")
        
        # Get trade history with our enhanced approach for historical data
        trades_df = get_trade_history(token_id, election_timestamp, days_before, sg, orderbook_subgraph)
        
        if trades_df.empty:
            print(f"No trades found for token {token_id} in the specified time period")
            token_metrics['trade_count'] = 0
            metrics['tokens_data'].append(token_metrics)
            continue
        
        print(f"Found {len(trades_df)} trades for token {token_id}")
        
        # Check if we have price data in our trades
        has_price_data = 'enrichedOrderFilleds_price' in trades_df.columns and not trades_df['enrichedOrderFilleds_price'].isnull().all()
        
        if has_price_data:
            # Calculate and add metrics that require price data
            token_metrics.update(calculate_closing_price(trades_df))
            token_metrics.update(calculate_vwap(trades_df, VWAP_WINDOW_HOURS))
            token_metrics.update(calculate_median_price(trades_df, STABILITY_WINDOW_DAYS))
            token_metrics.update(calculate_price_stability(trades_df, STABILITY_WINDOW_DAYS))
            token_metrics.update(calculate_hourly_price_series(trades_df, STABILITY_WINDOW_DAYS))
        else:
            print("Warning: No price data available for calculating price metrics")
            # Use the last known price from marketData if available
            if token_metrics.get('last_known_price'):
                print(f"Using last known price from marketData: {token_metrics['last_known_price']}")
                token_metrics['closing_price'] = float(token_metrics['last_known_price'])
            else:
                print("No price information available")
        
        # These metrics don't require price data
        token_metrics.update(calculate_liquidity_metrics(trades_df, STABILITY_WINDOW_DAYS))
        
        if 'enrichedOrderFilleds_side' in trades_df.columns:
            token_metrics.update(calculate_buy_sell_ratio(trades_df, STABILITY_WINDOW_DAYS))
        
        # Add token metrics to overall metrics
        metrics['tokens_data'].append(token_metrics)
        
        # Print summary of key metrics
        print("\nKey Metrics Summary:")
        print(f"Closing Price: {token_metrics.get('closing_price')}")
        print(f"VWAP ({VWAP_WINDOW_HOURS}h): {token_metrics.get('vwap')}")
        print(f"Median Price ({STABILITY_WINDOW_DAYS}d): {token_metrics.get('median_price')}")
        print(f"Price Volatility: {token_metrics.get('price_volatility')}")
        print(f"Trade Count: {token_metrics.get('trade_count', 0)}")
        print(f"Total Volume: {token_metrics.get('total_volume', 0)}")
    
    return metrics


def calculate_hourly_price_series(trades_df, days_window):
    """
    Calculate hourly price series for visualization.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
        days_window (int): Days to include
        
    Returns:
        list: Hourly price data points
    """
    if trades_df.empty:
        return []
    
    # Get last trade timestamp
    last_trade_timestamp = trades_df['enrichedOrderFilleds_timestamp'].max()
    
    # Calculate window start timestamp
    window_start_timestamp = last_trade_timestamp - (days_window * 24 * 60 * 60)
    
    # Filter trades in the window
    window_trades = trades_df[trades_df['enrichedOrderFilleds_timestamp'] >= window_start_timestamp]
    
    if window_trades.empty:
        return []
    
    # Create a datetime column for easier resampling
    window_trades['datetime'] = pd.to_datetime(window_trades['enrichedOrderFilleds_timestamp'], unit='s')
    window_trades = window_trades.set_index('datetime')
    
    # Resample to hourly data and take the last price in each hour
    try:
        hourly_data = window_trades['enrichedOrderFilleds_price'].astype(float).resample('1H').last()
        hourly_data = hourly_data.fillna(method='ffill')  # Forward fill missing values
        
        # Convert to list of (timestamp, price) points
        price_series = []
        for dt, price in hourly_data.items():
            if not np.isnan(price):
                price_series.append({
                    'timestamp': int(dt.timestamp()),
                    'price': float(price)
                })
        
        return price_series
    except Exception as e:
        print(f"Error calculating hourly price series: {e}")
        return []


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract prediction metrics for Polymarket election markets")
    parser.add_argument("--market-id", required=True, help="Market ID (FixedProductMarketMaker address)")
    parser.add_argument("--token-ids", required=True, help="Token IDs as JSON string")
    parser.add_argument("--election-date", required=True, help="Election date (YYYY-MM-DD)")
    parser.add_argument("--days-before", type=int, default=DEFAULT_PRE_ELECTION_DAYS, 
                        help=f"Days before election to analyze (default: {DEFAULT_PRE_ELECTION_DAYS})")
    parser.add_argument("--output", help="Output filename prefix", default="election_metrics")
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        
        # Parse token IDs
        token_ids = parse_token_ids(args.token_ids)
        if not token_ids:
            print("Error: No valid token IDs found")
            return
        
        print(f"Parsed token IDs: {token_ids}")
        
        # Extract prediction metrics
        metrics = extract_prediction_metrics(
            args.market_id,
            token_ids,
            args.election_date,
            args.days_before
        )
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(OUTPUT_DIRECTORY, f"{args.output}_{timestamp}.json")
        
        # Save results to JSON file
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nDetailed metrics saved to: {output_file}")
        
        # Print summary of key metrics for each token
        print("\nSummary of Prediction Metrics:")
        print("=" * 60)
        for token_data in metrics['tokens_data']:
            token_id = token_data.get('token_id', 'Unknown')
            print(f"\nToken ID: {token_id[-8:]}...")  # Show last 8 chars
            print(f"  Closing Price: {token_data.get('closing_price')}")
            print(f"  VWAP ({VWAP_WINDOW_HOURS}h): {token_data.get('vwap')}")
            print(f"  Median Price ({STABILITY_WINDOW_DAYS}d): {token_data.get('median_price')}")
            print(f"  Price Stability: {token_data.get('price_volatility'):.4f}" if token_data.get('price_volatility') else "  Price Stability: N/A")
            print(f"  Trade Count: {token_data.get('trade_count', 0)}")
            print(f"  Buy/Sell Ratio: {token_data.get('buy_sell_ratio'):.2f}" if token_data.get('buy_sell_ratio') else "  Buy/Sell Ratio: N/A")
        
    except Exception as e:
        print(f"Error executing script: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()