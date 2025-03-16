#!/usr/bin/env python3
"""
Simplified Polymarket Raw Data Collector

This script efficiently extracts comprehensive market data from Polymarket subgraphs including:
1. Trade data (enrichedOrderFilled events)
2. Orderbook entity data
3. OrdersMatchedEvent data - properly implemented

Usage:
    python market_data_collector.py --input top_election_markets.csv [options]

Options:
    --api-key
        Your Graph API key for accessing the subgraphs.
    --output-dir
        Directory to save the collected data.
    --token-col
        Column name containing token IDs in the input CSV.
    --batch-size
        Number of records to fetch per request.
    --timeout
        Timeout in seconds for each API request.
    --trades
        Collect trade data.
    --orderbook
        Collect orderbook entity data.
    --matched-events
        Collect OrdersMatchedEvent data.

Example:
    python market_data_collector.py --input top_election_markets.csv --api-key YOUR_API_KEY --output-dir polymarket_raw_data --trades --orderbook --matched-events

Note: The script uses the official Polymarket subgraph IDs and schema.

Author: Helen Wu
Last updated: 2025-03-15
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
from typing import Dict, List, Any, Optional, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_data_collection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

def make_api_request(url, headers, query, timeout=60, max_retries=5, retry_delay=3):
    """Make an API request with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=query, timeout=timeout)
            
            # Check for HTTP errors
            if response.status_code != 200:
                logger.warning(f"Request failed with status {response.status_code}: {response.text}")
                raise requests.exceptions.HTTPError(f"HTTP error {response.status_code}")
            
            # Parse response
            data = response.json()
            
            # Check for GraphQL errors
            if 'errors' in data:
                error_msg = '; '.join([err.get('message', 'Unknown error') for err in data.get('errors', [])])
                logger.warning(f"GraphQL errors: {error_msg}")
                if 'data' not in data or data['data'] is None:
                    raise Exception(f"GraphQL errors: {error_msg}")
            
            return data
            
        except (requests.exceptions.RequestException, 
                requests.exceptions.HTTPError, 
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                raise

def collect_token_trades(token_id, api_key, output_dir, batch_size=5000, timeout=60):
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
    progress = tqdm(desc=f"Token {token_id} trades", unit="batch", position=0, leave=True)
    
    try:
        # Keep querying until we get all trades
        while has_more:
            # Prepare GraphQL query with pagination
            query = {
                'query': f"""
                {{
                  enrichedOrderFilleds(
                    first: {batch_size}
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
            
            try:
                # Make request with retry
                data = make_api_request(url, headers, query, timeout=timeout)
                trades = data.get('data', {}).get('enrichedOrderFilleds', [])
                
                # Process the trades
                if trades:
                    # Add trades to our collection
                    all_trades.extend(trades)
                    
                    # Move to next batch
                    skip += len(trades)
                    progress.update(1)
                    
                    # Check if we've reached the end
                    if len(trades) < batch_size:
                        has_more = False
                else:
                    # No more trades
                    has_more = False
                
            except Exception as e:
                logger.error(f"Fatal error collecting trades for token {token_id}: {e}")
                has_more = False  # Exit the loop
        
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

def collect_orderbook_data(token_id, api_key, output_dir, timeout=60):
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
    
    # Prepare GraphQL query for orderbook data - expanded to include more details
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
    
    try:
        # Make request with retry
        data = make_api_request(url, headers, query, timeout=timeout)
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
    
    except Exception as e:
        logger.error(f"Error collecting orderbook data for token {token_id}: {e}")
        return {
            'token_id': token_id,
            'success': False,
            'error': str(e)
        }

def collect_orders_matched_events(token_id, api_key, output_dir, batch_size=1000, timeout=60):
    """
    Collect OrdersMatchedEvent data for a specific token and save to disk.
    Correctly implemented based on the actual schema.
    """
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
    
    # Track the unique event IDs we've already seen
    seen_event_ids = set()
    
    # Create progress bar
    progress = tqdm(desc=f"Token {token_id} matched events", unit="batch", position=0, leave=True)
    
    # First try to collect both ordersMatchedEvents and orderFilledEvents to be thorough
    try:
        # 1. First try OrdersMatchedEvent - looking for events where this token is the makerAssetID
        skip = 0
        has_more = True
        
        while has_more:
            # According to the schema, makerAssetID and takerAssetID are BigInt types
            query = {
                'query': f"""
                {{
                  ordersMatchedEvents(
                    first: {batch_size}
                    skip: {skip}
                    orderBy: timestamp
                    orderDirection: asc
                  ) {{
                    id
                    timestamp
                    makerAmountFilled
                    takerAmountFilled
                    makerAssetID
                    takerAssetID
                  }}
                }}
                """
            }
            
            try:
                data = make_api_request(url, headers, query, timeout=timeout)
                events = data.get('data', {}).get('ordersMatchedEvents', [])
                
                # Process the events - filter here for relevant token
                if events:
                    # Filter for events related to this token
                    filtered_events = []
                    for event in events:
                        maker_id = str(event.get('makerAssetID', ''))
                        taker_id = str(event.get('takerAssetID', ''))
                        if maker_id == token_id or taker_id == token_id:
                            # Add a source field to identify the event type
                            event['source'] = 'OrdersMatchedEvent'
                            filtered_events.append(event)
                    
                    # Only add events we haven't seen before
                    new_events_count = 0
                    for event in filtered_events:
                        if event['id'] not in seen_event_ids:
                            all_events.append(event)
                            seen_event_ids.add(event['id'])
                            new_events_count += 1
                    
                    # Move to next batch
                    skip += len(events)
                    progress.update(1)
                    
                    # Check if we've reached the end
                    if len(events) < batch_size or new_events_count == 0:
                        has_more = False
                else:
                    # No more events
                    has_more = False
            
            except Exception as e:
                logger.warning(f"Error collecting OrdersMatchedEvent makerAssetID events for token {token_id}: {e}")
                has_more = False  # Exit the loop
        
        # 2. Try OrdersMatchedEvent - looking for events where this token is the takerAssetID
        skip = 0
        has_more = True
        
        while has_more:
            query = {
                'query': f"""
                {{
                  ordersMatchedEvents(
                    first: {batch_size}
                    skip: {skip}
                    orderBy: timestamp
                    orderDirection: asc
                  ) {{
                    id
                    timestamp
                    makerAmountFilled
                    takerAmountFilled
                    makerAssetID
                    takerAssetID
                  }}
                }}
                """
            }
            
            try:
                data = make_api_request(url, headers, query, timeout=timeout)
                events = data.get('data', {}).get('ordersMatchedEvents', [])
                
                # Process the events
                if events:
                    # Only add events we haven't seen before
                    new_events_count = 0
                    for event in events:
                        if event['id'] not in seen_event_ids:
                            # Add a source field to identify the event type
                            event['source'] = 'OrdersMatchedEvent'
                            all_events.append(event)
                            seen_event_ids.add(event['id'])
                            new_events_count += 1
                    
                    # Move to next batch
                    skip += len(events)
                    progress.update(1)
                    
                    # Check if we've reached the end
                    if len(events) < batch_size or new_events_count == 0:
                        has_more = False
                else:
                    # No more events
                    has_more = False
            
            except Exception as e:
                logger.warning(f"Error collecting OrdersMatchedEvent takerAssetID events for token {token_id}: {e}")
                has_more = False  # Exit the loop
        
        # 3. Also try OrderFilledEvent as a backup - looking for events where this token is the makerAssetId
        skip = 0
        has_more = True
        
        while has_more:
            query = {
                'query': f"""
                {{
                  orderFilledEvents(
                    first: {batch_size}
                    skip: {skip}
                    orderBy: timestamp
                    orderDirection: asc
                    where: {{ 
                      makerAssetId: "{token_id}" 
                    }}
                  ) {{
                    id
                    timestamp
                    maker
                    taker
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
            
            try:
                data = make_api_request(url, headers, query, timeout=timeout)
                events = data.get('data', {}).get('orderFilledEvents', [])
                
                # Process the events
                if events:
                    # Only add events we haven't seen before
                    new_events_count = 0
                    for event in events:
                        if event['id'] not in seen_event_ids:
                            # Add a source field to identify the event type
                            event['source'] = 'OrderFilledEvent'
                            all_events.append(event)
                            seen_event_ids.add(event['id'])
                            new_events_count += 1
                    
                    # Move to next batch
                    skip += len(events)
                    progress.update(1)
                    
                    # Check if we've reached the end
                    if len(events) < batch_size or new_events_count == 0:
                        has_more = False
                else:
                    # No more events
                    has_more = False
            
            except Exception as e:
                logger.warning(f"Error collecting OrderFilledEvent makerAssetId events for token {token_id}: {e}")
                has_more = False  # Exit the loop
        
        # 4. Finally try OrderFilledEvent - looking for events where this token is the takerAssetId
        skip = 0
        has_more = True
        
        while has_more:
            query = {
                'query': f"""
                {{
                  orderFilledEvents(
                    first: {batch_size}
                    skip: {skip}
                    orderBy: timestamp
                    orderDirection: asc
                    where: {{ 
                      takerAssetId: "{token_id}" 
                    }}
                  ) {{
                    id
                    timestamp
                    maker
                    taker
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
            
            try:
                data = make_api_request(url, headers, query, timeout=timeout)
                events = data.get('data', {}).get('orderFilledEvents', [])
                
                # Process the events
                if events:
                    # Only add events we haven't seen before
                    new_events_count = 0
                    for event in events:
                        if event['id'] not in seen_event_ids:
                            # Add a source field to identify the event type
                            event['source'] = 'OrderFilledEvent'
                            all_events.append(event)
                            seen_event_ids.add(event['id'])
                            new_events_count += 1
                    
                    # Move to next batch
                    skip += len(events)
                    progress.update(1)
                    
                    # Check if we've reached the end
                    if len(events) < batch_size or new_events_count == 0:
                        has_more = False
                else:
                    # No more events
                    has_more = False
            
            except Exception as e:
                logger.warning(f"Error collecting OrderFilledEvent takerAssetId events for token {token_id}: {e}")
                has_more = False  # Exit the loop
        
        # Close progress bar
        progress.close()
        
        # Process results
        if all_events:
            logger.info(f"Collected {len(all_events)} total events for token {token_id}")
            
            try:
                # Group events by source for processing
                ordered_matched_events = [e for e in all_events if e['source'] == 'OrdersMatchedEvent']
                order_filled_events = [e for e in all_events if e['source'] == 'OrderFilledEvent']
                
                # Log the breakdown
                logger.info(f"  - OrdersMatchedEvent: {len(ordered_matched_events)}")
                logger.info(f"  - OrderFilledEvent: {len(order_filled_events)}")
                
                # Create two separate dataframes and then combine them
                dfs = []
                
                # Process OrdersMatchedEvents
                if ordered_matched_events:
                    # OrdersMatchedEvent has different structure
                    matched_df = pd.DataFrame(ordered_matched_events)
                    matched_df['event_type'] = 'OrdersMatchedEvent'
                    
                    # Convert numeric columns
                    numeric_cols = ['makerAmountFilled', 'takerAmountFilled', 'makerAssetID', 'takerAssetID', 'timestamp']
                    for col in numeric_cols:
                        if col in matched_df.columns:
                            matched_df[col] = pd.to_numeric(matched_df[col], errors='coerce')
                    
                    # Rename columns for consistency with OrderFilledEvent
                    matched_df = matched_df.rename(columns={
                        'makerAssetID': 'makerAssetId',
                        'takerAssetID': 'takerAssetId'
                    })
                    
                    dfs.append(matched_df)
                
                # Process OrderFilledEvents
                if order_filled_events:
                    # OrderFilledEvent has different structure
                    filled_df = pd.DataFrame(order_filled_events)
                    filled_df['event_type'] = 'OrderFilledEvent'
                    
                    # Convert numeric columns
                    numeric_cols = ['makerAmountFilled', 'takerAmountFilled', 'fee', 'timestamp']
                    for col in numeric_cols:
                        if col in filled_df.columns:
                            filled_df[col] = pd.to_numeric(filled_df[col], errors='coerce')
                    
                    dfs.append(filled_df)
                
                # Combine the dataframes if we have more than one
                if len(dfs) > 1:
                    # Get common columns for the join
                    common_cols = set(dfs[0].columns)
                    for df in dfs[1:]:
                        common_cols = common_cols.intersection(set(df.columns))
                    
                    # Use concat with only common columns
                    events_df = pd.concat([df[list(common_cols)] for df in dfs], ignore_index=True)
                else:
                    events_df = dfs[0]
                
                # Add token ID for reference
                events_df['token_id'] = token_id
                
                # Save to parquet file
                events_df.to_parquet(output_file, compression='snappy')
                
                logger.info(f"Saved events to {output_file}")
                
                return {
                    'token_id': token_id,
                    'event_count': len(all_events),
                    'matched_events_count': len(ordered_matched_events),
                    'filled_events_count': len(order_filled_events),
                    'file_path': output_file,
                    'success': True
                }
            except Exception as e:
                logger.error(f"Error processing events data for token {token_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    'token_id': token_id,
                    'success': False,
                    'error': str(e)
                }
        else:
            logger.info(f"No events found for token {token_id}")
            
            # Create an empty DataFrame and save it to maintain consistency
            empty_df = pd.DataFrame(columns=['id', 'timestamp', 'event_type', 'token_id'])
            empty_df['token_id'] = token_id
            empty_df.to_parquet(output_file, compression='snappy')
            
            return {
                'token_id': token_id,
                'event_count': 0,
                'success': True
            }
    
    except Exception as e:
        logger.error(f"Error collecting events for token {token_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'token_id': token_id,
            'success': False,
            'error': str(e)
        }

def process_token(token_id, api_key, output_dirs, collect_types, config):
    """Process a single token with appropriate error handling, collecting all specified data types."""
    results = {
        'token_id': token_id,
        'success': True,
        'trade_data': None,
        'orderbook_data': None,
        'matched_events_data': None,
        'timing': {}
    }
    
    try:
        # Collect trade data if requested
        if 'trades' in collect_types:
            start_time = time.time()
            trade_result = collect_token_trades(
                token_id, 
                api_key, 
                output_dirs['trades'],
                batch_size=config['batch_size'],
                timeout=config['timeout']
            )
            results['trade_data'] = trade_result
            results['timing']['trades'] = time.time() - start_time
            if not trade_result.get('success', False):
                results['success'] = False
        
        # Collect orderbook data if requested
        if 'orderbooks' in collect_types:
            start_time = time.time()
            orderbook_result = collect_orderbook_data(
                token_id, 
                api_key, 
                output_dirs['orderbooks'],
                timeout=config['timeout']
            )
            results['orderbook_data'] = orderbook_result
            results['timing']['orderbooks'] = time.time() - start_time
            if not orderbook_result.get('success', False):
                results['success'] = False
        
        # Collect orders matched events if requested
        if 'matched_events' in collect_types:
            start_time = time.time()
            events_result = collect_orders_matched_events(
                token_id, 
                api_key, 
                output_dirs['matched_events'],
                batch_size=config['batch_size'],
                timeout=config['timeout']
            )
            results['matched_events_data'] = events_result
            results['timing']['matched_events'] = time.time() - start_time
            if not events_result.get('success', False):
                results['success'] = False
        
        return results
    
    except Exception as e:
        logger.error(f"Unhandled error processing token {token_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results['success'] = False
        results['error'] = str(e)
        return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Polymarket Raw Data Collector")
    parser.add_argument("--input", required=True, help="Input CSV file with market data")
    parser.add_argument("--column", default="clobTokenIds", help="Column name with token IDs (default: clobTokenIds)")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens to process")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum parallel workers (default: 8)")
    parser.add_argument("--sequential", action="store_true", help="Process tokens sequentially (no parallelism)")
    parser.add_argument("--skip-trades", action="store_true", help="Skip trade data collection")
    parser.add_argument("--skip-orderbooks", action="store_true", help="Skip orderbook data collection")
    parser.add_argument("--skip-matched-events", action="store_true", help="Skip matched events collection")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds (default: 60)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for queries (default: 1000)")
    parser.add_argument("--output-dir", default="polymarket_raw_data", help="Base output directory")
    args = parser.parse_args()
    
    # Configuration dictionary
    config = {
        'timeout': args.timeout,
        'batch_size': args.batch_size,
        'max_workers': args.max_workers,
        'data_dir': args.output_dir,
    }
    
    # Create output directory paths
    output_dirs = {
        'base': config['data_dir'],
        'trades': os.path.join(config['data_dir'], "trades"),
        'orderbooks': os.path.join(config['data_dir'], "orderbooks"),
        'matched_events': os.path.join(config['data_dir'], "matched_events")
    }
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        logger.error("GRAPH_API_KEY not found in environment variables")
        return
    
    # Create output directories
    try:
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Output directory ready: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return
    
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
    
    # Extract token IDs from CSV
    tokens, market_tokens, market_id_to_question = extract_unique_tokens(args.input, args.column)
    
    if not tokens:
        logger.error("No token IDs found in CSV. Please check the file and column name.")
        return
    
    # Limit to max-tokens if specified
    if args.max_tokens and len(tokens) > args.max_tokens:
        logger.info(f"Limiting to {args.max_tokens} tokens (out of {len(tokens)} total)")
        tokens = tokens[:args.max_tokens]
    
    # Save market-to-tokens mapping for reference
    with open(os.path.join(output_dirs['base'], "market_tokens.json"), 'w') as f:
        json.dump(market_tokens, f, indent=2)
    
    # Save market ID to question mapping for reference
    with open(os.path.join(output_dirs['base'], "market_id_to_question.json"), 'w') as f:
        json.dump(market_id_to_question, f, indent=2)
    
    # Process tokens
    start_time = time.time()
    
    if args.sequential:
        # Process sequentially
        logger.info(f"Processing {len(tokens)} tokens sequentially")
        results = []
        for token_id in tqdm(tokens, desc="Processing tokens", position=0, leave=True):
            result = process_token(token_id, api_key, output_dirs, collect_types, config)
            results.append(result)
    else:
        # Process in parallel
        workers = min(config['max_workers'], len(tokens))
        logger.info(f"Processing {len(tokens)} tokens in parallel with {workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_token, token_id, api_key, output_dirs, collect_types, config) 
                      for token_id in tokens]
            
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tokens", position=0, leave=True):
                results.append(future.result())
    
    total_time = time.time() - start_time
    
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
    
    # Calculate timing statistics
    timing_stats = {
        'trades': {'total': 0, 'count': 0, 'avg': 0},
        'orderbooks': {'total': 0, 'count': 0, 'avg': 0},
        'matched_events': {'total': 0, 'count': 0, 'avg': 0},
    }
    
    for result in results:
        timing = result.get('timing', {})
        for data_type, time_taken in timing.items():
            if data_type in timing_stats:
                timing_stats[data_type]['total'] += time_taken
                timing_stats[data_type]['count'] += 1
    
    # Calculate averages
    for data_type in timing_stats:
        if timing_stats[data_type]['count'] > 0:
            timing_stats[data_type]['avg'] = timing_stats[data_type]['total'] / timing_stats[data_type]['count']
    
    logger.info(f"Processing complete: {successful}/{len(results)} tokens successful in {total_time:.2f} seconds")
    
    if 'trades' in collect_types:
        logger.info(f"Trade data collection: {trade_successes}/{len(trade_results)} successful ({trade_skipped} skipped)")
        logger.info(f"Tokens with trades: {trade_with_data}/{len(trade_results)}")
        logger.info(f"Total trades collected: {total_trades:,}")
        if timing_stats['trades']['count'] > 0:
            logger.info(f"Average time per token: {timing_stats['trades']['avg']:.2f} seconds")
    
    if 'orderbooks' in collect_types:
        logger.info(f"Orderbook data collection: {orderbook_successes}/{len(orderbook_results)} successful ({orderbook_skipped} skipped)")
        logger.info(f"Tokens with orderbook data: {orderbook_with_data}/{len(orderbook_results)}")
        if timing_stats['orderbooks']['count'] > 0:
            logger.info(f"Average time per token: {timing_stats['orderbooks']['avg']:.2f} seconds")
    
    if 'matched_events' in collect_types:
        logger.info(f"Matched events collection: {events_successes}/{len(events_results)} successful ({events_skipped} skipped)")
        logger.info(f"Tokens with matched events: {events_with_data}/{len(events_results)}")
        logger.info(f"Total matched events collected: {total_events:,}")
        if timing_stats['matched_events']['count'] > 0:
            logger.info(f"Average time per token: {timing_stats['matched_events']['avg']:.2f} seconds")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'tokens_processed': len(results),
        'tokens_successful': successful,
        'total_time_seconds': total_time,
        'collected_data_types': collect_types,
        'settings': {
            'batch_size': config['batch_size'],
            'max_workers': workers if not args.sequential else 1,
            'timeout': config['timeout']
        },
        'timing_stats': timing_stats,
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
    
    # Save token-level results
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
    
    # Save with timestamp
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(output_dirs['base'], f"collection_summary_{timestamp_str}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save as the standard file name for backward compatibility
    with open(os.path.join(output_dirs['base'], "collection_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {summary_file}")
    logger.info(f"Processing completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()