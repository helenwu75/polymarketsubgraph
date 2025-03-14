#!/usr/bin/env python3
"""
Enhanced Polymarket Raw Data Collector

This script efficiently extracts comprehensive market data from Polymarket subgraphs including:
1. Trade data (enrichedOrderFilled events) - unchanged from original
2. Orderbook entity data - NEW
3. OrdersMatchedEvent data - NEW

Usage:
    python enhanced_market_collector.py --input top_election_markets.csv [options]
"""

import os
import sys
import json
import ast
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_data_collection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 5000
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_WORKERS = 8  # Number of parallel workers

# Output directories
DATA_DIR = "polymarket_raw_data"
TRADES_DIR = f"{DATA_DIR}/trades"
ORDERBOOKS_DIR = f"{DATA_DIR}/orderbooks"
MATCHED_EVENTS_DIR = f"{DATA_DIR}/matched_events"

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

def extract_unique_tokens(csv_file, token_col='clobTokenIds'):
    """Extract all unique token IDs from a CSV file."""
    logger.info(f"Extracting unique tokens from {csv_file}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        if token_col not in df.columns:
            raise ValueError(f"Column '{token_col}' not found in CSV")
        
        # Extract all token IDs
        all_tokens = set()
        market_tokens = {}
        market_id_to_question = {}
        
        for i, row in df.iterrows():
            token_ids = parse_token_ids(row[token_col])
            all_tokens.update(token_ids)
            
            # Keep track of which tokens belong to which markets
            if token_ids:
                question = row.get('question', f"Market_{i}")
                market_id = row.get('marketMakerAddress', None)
                
                if market_id:
                    market_tokens[market_id] = token_ids
                    market_id_to_question[market_id] = question
                else:
                    # Use question as fallback identifier
                    market_tokens[question] = token_ids
        
        # Remove any empty strings
        all_tokens = {t for t in all_tokens if t}
        
        logger.info(f"Found {len(all_tokens)} unique tokens across {len(market_tokens)} markets")
        return list(all_tokens), market_tokens, market_id_to_question
    
    except Exception as e:
        logger.error(f"Error extracting tokens: {e}")
        return [], {}, {}

def collect_token_trades(token_id, api_key, output_dir):
    """Collect all trade data for a specific token and save to disk."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Check if we already have data for this token
    output_file = os.path.join(output_dir, f"{token_id}.parquet")
    if os.path.exists(output_file):
        logger.info(f"Trade data for token {token_id} already exists, skipping")
        # Return stats from the existing file to include in the summary
        try:
            existing_df = pd.read_parquet(output_file)
            return {
                'token_id': token_id,
                'trade_count': len(existing_df),
                'file_path': output_file,
                'success': True,
                'skipped': True
            }
        except Exception as e:
            logger.warning(f"Error reading existing file for {token_id}: {e}")
            # Continue with collection as the file might be corrupted
    
    logger.info(f"Collecting trades for token {token_id}")
    
    # Prepare for pagination
    all_trades = []
    skip = 0
    has_more = True
    
    # Create progress bar
    progress = tqdm(desc=f"Token {token_id}", unit="batch")
    
    try:
        # Keep querying until we get all trades
        while has_more:
            # Prepare GraphQL query with pagination
            query = {
                'query': f"""
                {{
                  enrichedOrderFilleds(
                    first: {BATCH_SIZE}
                    skip: {skip}
                    orderBy: timestamp
                    orderDirection: asc
                    where: {{ market: "{token_id}" }}
                  ) {{
                    id
                    timestamp
                    price
                    side
                    size
                    maker {{ id }}
                    taker {{ id }}
                    transactionHash
                  }}
                }}
                """
            }
            
            # Try the request with retries
            for retry in range(MAX_RETRIES):
                try:
                    response = requests.post(url, headers=headers, json=query, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        trades = data.get('data', {}).get('enrichedOrderFilleds', [])
                        
                        # Process the trades
                        if trades:
                            # Add trades to our collection
                            all_trades.extend(trades)
                            
                            # Move to next batch
                            skip += len(trades)
                            progress.update(1)
                            
                            # Check if we've reached the end
                            if len(trades) < BATCH_SIZE:
                                has_more = False
                                break
                        else:
                            # No more trades
                            has_more = False
                            break
                        
                        # Success - break retry loop
                        break
                        
                    else:
                        logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                        if retry < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                        else:
                            raise Exception(f"Failed after {MAX_RETRIES} retries")
                
                except Exception as e:
                    logger.warning(f"Error on retry {retry+1}/{MAX_RETRIES}: {e}")
                    if retry < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        raise
        
        # Close progress bar
        progress.close()
        
        # Process results
        if all_trades:
            logger.info(f"Collected {len(all_trades)} trades for token {token_id}")
            
            # Convert to pandas DataFrame for easier processing
            trades_df = pd.json_normalize(all_trades)
            
            # Add token ID for reference
            trades_df['token_id'] = token_id
            
            # Rename columns for clarity
            if 'maker.id' in trades_df.columns:
                trades_df.rename(columns={'maker.id': 'maker_id'}, inplace=True)
            if 'taker.id' in trades_df.columns:
                trades_df.rename(columns={'taker.id': 'taker_id'}, inplace=True)
            
            # Save to parquet file (more efficient than CSV)
            trades_df.to_parquet(output_file, compression='snappy')
            
            logger.info(f"Saved trades to {output_file}")
            
            return {
                'token_id': token_id,
                'trade_count': len(all_trades),
                'file_path': output_file,
                'success': True
            }
        else:
            logger.info(f"No trades found for token {token_id}")
            return {
                'token_id': token_id,
                'trade_count': 0,
                'success': True
            }
    
    except Exception as e:
        logger.error(f"Error collecting trades for token {token_id}: {e}")
        return {
            'token_id': token_id,
            'success': False,
            'error': str(e)
        }

def collect_orderbook_data(token_id, api_key, output_dir):
    """Collect orderbook entity data for a specific token and save to disk."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Check if we already have data for this token
    output_file = os.path.join(output_dir, f"{token_id}.json")
    if os.path.exists(output_file):
        logger.info(f"Orderbook data for token {token_id} already exists, skipping")
        # Return success with skipped flag
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            return {
                'token_id': token_id,
                'file_path': output_file,
                'success': True,
                'skipped': True,
                'data': existing_data
            }
        except Exception as e:
            logger.warning(f"Error reading existing orderbook file for {token_id}: {e}")
            # Continue with collection as the file might be corrupted
    
    logger.info(f"Collecting orderbook data for token {token_id}")
    
    # Prepare GraphQL query for orderbook data
    query = {
        'query': f"""
        {{
          orderbook(id: "{token_id}") {{
            id
            tradesQuantity
            buysQuantity
            sellsQuantity
            collateralVolume
            scaledCollateralVolume
            collateralBuyVolume
            scaledCollateralBuyVolume
            collateralSellVolume
            scaledCollateralSellVolume
            lastActiveDay
          }}
        }}
        """
    }
    
    # Try the request with retries
    for retry in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                orderbook_data = data.get('data', {}).get('orderbook')
                
                if orderbook_data:
                    logger.info(f"Found orderbook data for token {token_id}")
                    
                    # Save to JSON file
                    with open(output_file, 'w') as f:
                        json.dump(orderbook_data, f, indent=2)
                    
                    logger.info(f"Saved orderbook data to {output_file}")
                    
                    return {
                        'token_id': token_id,
                        'file_path': output_file,
                        'success': True,
                        'data': orderbook_data
                    }
                else:
                    logger.info(f"No orderbook data found for token {token_id}")
                    return {
                        'token_id': token_id,
                        'success': True,
                        'data': None
                    }
            
            else:
                logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                if retry < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise Exception(f"Failed after {MAX_RETRIES} retries")
        
        except Exception as e:
            logger.warning(f"Error on retry {retry+1}/{MAX_RETRIES}: {e}")
            if retry < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
    
    # If we reached here, all retries failed
    logger.error(f"Failed to collect orderbook data for token {token_id} after all retries")
    return {
        'token_id': token_id,
        'success': False,
        'error': "Failed after all retries"
    }

def collect_orders_matched_events(token_id, api_key, output_dir):
    """Collect OrdersMatchedEvent data for a specific token and save to disk."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Check if we already have data for this token
    output_file = os.path.join(output_dir, f"{token_id}.parquet")
    if os.path.exists(output_file):
        logger.info(f"OrdersMatchedEvent data for token {token_id} already exists, skipping")
        # Return stats from the existing file
        try:
            existing_df = pd.read_parquet(output_file)
            return {
                'token_id': token_id,
                'event_count': len(existing_df),
                'file_path': output_file,
                'success': True,
                'skipped': True
            }
        except Exception as e:
            logger.warning(f"Error reading existing matched events file for {token_id}: {e}")
            # Continue with collection as the file might be corrupted
    
    logger.info(f"Collecting orders matched events for token {token_id}")
    
    # Prepare for pagination
    all_events = []
    skip = 0
    has_more = True
    
    # Create progress bar
    progress = tqdm(desc=f"Token {token_id} events", unit="batch")
    
    try:
        # Keep querying until we get all events
        while has_more:
            # Prepare GraphQL query with pagination
            # Note: The OrdersMatchedEvent query needs to be adjusted based on the actual schema
            # We're querying by makerAssetID or takerAssetID to find relevant events
            query = {
                'query': f"""
                {{
                  orderFilledEvents(
                    first: {BATCH_SIZE}
                    skip: {skip}
                    orderBy: timestamp
                    orderDirection: asc
                    where: {{ 
                      makerAssetId: "{token_id}" 
                    }}
                  ) {{
                    id
                    timestamp
                    maker {{ id }}
                    taker {{ id }}
                    makerAmountFilled
                    takerAmountFilled
                    makerAssetId
                    takerAssetId
                    fee
                    transactionHash
                    orderHash
                  }}
                }}
                """
            }
            
            # Try the request with retries
            for retry in range(MAX_RETRIES):
                try:
                    response = requests.post(url, headers=headers, json=query, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        events = data.get('data', {}).get('orderFilledEvents', [])
                        
                        # Process the events
                        if events:
                            # Add events to our collection
                            all_events.extend(events)
                            
                            # Move to next batch
                            skip += len(events)
                            progress.update(1)
                            
                            # Check if we've reached the end
                            if len(events) < BATCH_SIZE:
                                has_more = False
                                break
                        else:
                            # No more events
                            has_more = False
                            break
                        
                        # Success - break retry loop
                        break
                        
                    else:
                        logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                        if retry < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                        else:
                            raise Exception(f"Failed after {MAX_RETRIES} retries")
                
                except Exception as e:
                    logger.warning(f"Error on retry {retry+1}/{MAX_RETRIES}: {e}")
                    if retry < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        raise
        
        # Also check events where this token is the taker asset
        skip = 0
        has_more = True
        
        while has_more:
            query = {
                'query': f"""
                {{
                  orderFilledEvents(
                    first: {BATCH_SIZE}
                    skip: {skip}
                    orderBy: timestamp
                    orderDirection: asc
                    where: {{ 
                      takerAssetId: "{token_id}" 
                    }}
                  ) {{
                    id
                    timestamp
                    maker {{ id }}
                    taker {{ id }}
                    makerAmountFilled
                    takerAmountFilled
                    makerAssetId
                    takerAssetId
                    fee
                    transactionHash
                    orderHash
                  }}
                }}
                """
            }
            
            # Try the request with retries
            for retry in range(MAX_RETRIES):
                try:
                    response = requests.post(url, headers=headers, json=query, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        events = data.get('data', {}).get('orderFilledEvents', [])
                        
                        # Process the events
                        if events:
                            # Add events to our collection (checking for duplicates)
                            event_ids = {e['id'] for e in all_events}
                            new_events = [e for e in events if e['id'] not in event_ids]
                            all_events.extend(new_events)
                            
                            # Move to next batch
                            skip += len(events)
                            progress.update(1)
                            
                            # Check if we've reached the end
                            if len(events) < BATCH_SIZE:
                                has_more = False
                                break
                        else:
                            # No more events
                            has_more = False
                            break
                        
                        # Success - break retry loop
                        break
                        
                    else:
                        logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                        if retry < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                        else:
                            raise Exception(f"Failed after {MAX_RETRIES} retries")
                
                except Exception as e:
                    logger.warning(f"Error on retry {retry+1}/{MAX_RETRIES}: {e}")
                    if retry < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        raise
        
        # Close progress bar
        progress.close()
        
        # Process results
        if all_events:
            logger.info(f"Collected {len(all_events)} matched events for token {token_id}")
            
            # Convert to pandas DataFrame for easier processing
            events_df = pd.json_normalize(all_events)
            
            # Add token ID for reference
            events_df['token_id'] = token_id
            
            # Rename columns for clarity
            if 'maker.id' in events_df.columns:
                events_df.rename(columns={'maker.id': 'maker_id'}, inplace=True)
            if 'taker.id' in events_df.columns:
                events_df.rename(columns={'taker.id': 'taker_id'}, inplace=True)
            
            # Save to parquet file
            events_df.to_parquet(output_file, compression='snappy')
            
            logger.info(f"Saved matched events to {output_file}")
            
            return {
                'token_id': token_id,
                'event_count': len(all_events),
                'file_path': output_file,
                'success': True
            }
        else:
            logger.info(f"No matched events found for token {token_id}")
            return {
                'token_id': token_id,
                'event_count': 0,
                'success': True
            }
    
    except Exception as e:
        logger.error(f"Error collecting matched events for token {token_id}: {e}")
        return {
            'token_id': token_id,
            'success': False,
            'error': str(e)
        }

def process_token(token_id, api_key, output_dirs, collect_types):
    """Process a single token with appropriate error handling, collecting all specified data types."""
    results = {
        'token_id': token_id,
        'success': True,
        'trade_data': None,
        'orderbook_data': None,
        'matched_events_data': None
    }
    
    try:
        # Collect trade data if requested
        if 'trades' in collect_types:
            trade_result = collect_token_trades(token_id, api_key, output_dirs['trades'])
            results['trade_data'] = trade_result
            if not trade_result.get('success', False):
                results['success'] = False
        
        # Collect orderbook data if requested
        if 'orderbooks' in collect_types:
            orderbook_result = collect_orderbook_data(token_id, api_key, output_dirs['orderbooks'])
            results['orderbook_data'] = orderbook_result
            if not orderbook_result.get('success', False):
                results['success'] = False
        
        # Collect orders matched events if requested
        if 'matched_events' in collect_types:
            events_result = collect_orders_matched_events(token_id, api_key, output_dirs['matched_events'])
            results['matched_events_data'] = events_result
            if not events_result.get('success', False):
                results['success'] = False
        
        return results
    
    except Exception as e:
        logger.error(f"Unhandled error processing token {token_id}: {e}")
        results['success'] = False
        results['error'] = str(e)
        return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Polymarket Raw Data Collector")
    parser.add_argument("--input", required=True, help="Input CSV file with market data")
    parser.add_argument("--column", default="clobTokenIds", help="Column name with token IDs (default: clobTokenIds)")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens to process")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help=f"Maximum parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--sequential", action="store_true", help="Process tokens sequentially (no parallelism)")
    parser.add_argument("--skip-trades", action="store_true", help="Skip trade data collection")
    parser.add_argument("--skip-orderbooks", action="store_true", help="Skip orderbook data collection")
    parser.add_argument("--skip-matched-events", action="store_true", help="Skip matched events collection")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        logger.error("GRAPH_API_KEY not found in environment variables")
        return
    
    # Create output directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TRADES_DIR, exist_ok=True)
    os.makedirs(ORDERBOOKS_DIR, exist_ok=True)
    os.makedirs(MATCHED_EVENTS_DIR, exist_ok=True)
    
    # Determine which data types to collect
    collect_types = []
    if not args.skip_trades:
        collect_types.append('trades')
    if not args.skip_orderbooks:
        collect_types.append('orderbooks')
    if not args.skip_matched_events:
        collect_types.append('matched_events')
    
    if not collect_types:
        logger.error("No data types selected for collection (all are skipped)")
        return
    
    logger.info(f"Will collect the following data types: {', '.join(collect_types)}")
    
    # Set up output directory mapping
    output_dirs = {
        'trades': TRADES_DIR,
        'orderbooks': ORDERBOOKS_DIR,
        'matched_events': MATCHED_EVENTS_DIR
    }
    
    # Extract token IDs from CSV
    tokens, market_tokens, market_id_to_question = extract_unique_tokens(args.input, args.column)
    
    if not tokens:
        logger.error("No token IDs found in CSV. Please check the file and column name.")
        return
    
    # Limit to max-tokens if specified
    if args.max_tokens and len(tokens) > args.max_tokens:
        logger.info(f"Limiting to {args.max_tokens} tokens")
        tokens = tokens[:args.max_tokens]
    
    # Save market-to-tokens mapping for reference
    with open(os.path.join(DATA_DIR, "market_tokens.json"), 'w') as f:
        json.dump(market_tokens, f, indent=2)
    
    # Save market ID to question mapping for reference
    with open(os.path.join(DATA_DIR, "market_id_to_question.json"), 'w') as f:
        json.dump(market_id_to_question, f, indent=2)
    
    # Process tokens
    if args.sequential:
        # Process sequentially
        logger.info(f"Processing {len(tokens)} tokens sequentially")
        results = []
        for token_id in tqdm(tokens, desc="Processing tokens"):
            result = process_token(token_id, api_key, output_dirs, collect_types)
            results.append(result)
    else:
        # Process in parallel
        workers = min(args.max_workers, len(tokens))
        logger.info(f"Processing {len(tokens)} tokens in parallel with {workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_token, token_id, api_key, output_dirs, collect_types) 
                      for token_id in tokens]
            
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tokens"):
                results.append(future.result())
    
    # Summarize results
    successful = sum(1 for r in results if r.get('success', False))
    
    # Summarize trade data collection
    trade_results = [r.get('trade_data') for r in results if r.get('trade_data')]
    trade_successes = sum(1 for r in trade_results if r.get('success', False))
    trade_skipped = sum(1 for r in trade_results if r.get('skipped', False))
    trade_with_data = sum(1 for r in trade_results if r.get('trade_count', 0) > 0)
    total_trades = sum(r.get('trade_count', 0) for r in trade_results if r.get('success', False))
    
    # Summarize orderbook data collection
    orderbook_results = [r.get('orderbook_data') for r in results if r.get('orderbook_data')]
    orderbook_successes = sum(1 for r in orderbook_results if r.get('success', False))
    orderbook_skipped = sum(1 for r in orderbook_results if r.get('skipped', False))
    orderbook_with_data = sum(1 for r in orderbook_results if r.get('data') is not None)
    
    # Summarize matched events data collection
    events_results = [r.get('matched_events_data') for r in results if r.get('matched_events_data')]
    events_successes = sum(1 for r in events_results if r.get('success', False))
    events_skipped = sum(1 for r in events_results if r.get('skipped', False))
    events_with_data = sum(1 for r in events_results if r.get('event_count', 0) > 0)
    total_events = sum(r.get('event_count', 0) for r in events_results if r.get('success', False))
    
    logger.info(f"Processing complete: {successful}/{len(results)} tokens successful")
    
    if 'trades' in collect_types:
        logger.info(f"Trade data collection: {trade_successes}/{len(trade_results)} successful ({trade_skipped} skipped)")
        logger.info(f"Tokens with trades: {trade_with_data}/{len(trade_results)}")
        logger.info(f"Total trades collected: {total_trades:,}")
    
    if 'orderbooks' in collect_types:
        logger.info(f"Orderbook data collection: {orderbook_successes}/{len(orderbook_results)} successful ({orderbook_skipped} skipped)")
        logger.info(f"Tokens with orderbook data: {orderbook_with_data}/{len(orderbook_results)}")
    
    if 'matched_events' in collect_types:
        logger.info(f"Matched events collection: {events_successes}/{len(events_results)} successful ({events_skipped} skipped)")
        logger.info(f"Tokens with matched events: {events_with_data}/{len(events_results)}")
        logger.info(f"Total matched events collected: {total_events:,}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'tokens_processed': len(results),
        'tokens_successful': successful,
        'collected_data_types': collect_types,
        'trade_summary': {
            'total_tokens': len(trade_results),
            'successful_tokens': trade_successes,
            'skipped_tokens': trade_skipped,
            'tokens_with_data': trade_with_data,
            'total_trades': total_trades
        },
        'orderbook_summary': {
            'total_tokens': len(orderbook_results),
            'successful_tokens': orderbook_successes,
            'skipped_tokens': orderbook_skipped,
            'tokens_with_data': orderbook_with_data
        },
        'matched_events_summary': {
            'total_tokens': len(events_results),
            'successful_tokens': events_successes,
            'skipped_tokens': events_skipped,
            'tokens_with_data': events_with_data,
            'total_events': total_events
        }
    }
    
    # Save detailed token results separately to avoid huge JSON file
    token_results = []
    for result in results:
        # Create a simplified version without nested data structures
        simple_result = {
            'token_id': result['token_id'],
            'success': result['success']
        }
        
        # Add trade summary if available
        if result.get('trade_data'):
            simple_result['trade_success'] = result['trade_data'].get('success', False)
            simple_result['trade_count'] = result['trade_data'].get('trade_count', 0)
            simple_result['trade_skipped'] = result['trade_data'].get('skipped', False)
        
        # Add orderbook summary if available
        if result.get('orderbook_data'):
            simple_result['orderbook_success'] = result['orderbook_data'].get('success', False)
            simple_result['orderbook_skipped'] = result['orderbook_data'].get('skipped', False)
            simple_result['has_orderbook_data'] = result['orderbook_data'].get('data') is not None
        
        # Add matched events summary if available
        if result.get('matched_events_data'):
            simple_result['events_success'] = result['matched_events_data'].get('success', False)
            simple_result['events_count'] = result['matched_events_data'].get('event_count', 0)
            simple_result['events_skipped'] = result['matched_events_data'].get('skipped', False)
        
        token_results.append(simple_result)
    
    # Add token results to summary
    summary['token_results'] = token_results
    
    with open(os.path.join(DATA_DIR, "enhanced_collection_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {DATA_DIR}/enhanced_collection_summary.json")

if __name__ == "__main__":
    main()