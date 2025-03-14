#!/usr/bin/env python3
"""
Polymarket Raw Data Collector

This script efficiently extracts raw trade data for Polymarket tokens without 
performing any calculations. It focuses on collecting data that can be analyzed later.

Usage:
    python polymarket_raw_collector.py --input top_election_markets.csv [options]
"""

import os
import sys
import json
import ast
import time
import argparse
from datetime import datetime
import pandas as pd
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
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_WORKERS = 8  # Number of parallel workers

# Output directories
DATA_DIR = "polymarket_raw_data"
TRADES_DIR = f"{DATA_DIR}/trades"

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
        
        for i, row in df.iterrows():
            token_ids = parse_token_ids(row[token_col])
            all_tokens.update(token_ids)
            
            # Also keep track of which tokens belong to which markets
            if token_ids:
                question = row.get('question', f"Market_{i}")
                market_tokens[question] = token_ids
        
        # Remove any empty strings
        all_tokens = {t for t in all_tokens if t}
        
        logger.info(f"Found {len(all_tokens)} unique tokens across {len(market_tokens)} markets")
        return list(all_tokens), market_tokens
    
    except Exception as e:
        logger.error(f"Error extracting tokens: {e}")
        return [], {}

def collect_token_trades(token_id, api_key, output_dir):
    """Collect all trade data for a specific token and save to disk."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
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
            output_file = os.path.join(output_dir, f"{token_id}.parquet")
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

def process_token(token_id, api_key, output_dir):
    """Process a single token with appropriate error handling."""
    try:
        return collect_token_trades(token_id, api_key, output_dir)
    except Exception as e:
        logger.error(f"Unhandled error processing token {token_id}: {e}")
        return {
            'token_id': token_id,
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Polymarket Raw Data Collector")
    parser.add_argument("--input", required=True, help="Input CSV file with market data")
    parser.add_argument("--column", default="clobTokenIds", help="Column name with token IDs (default: clobTokenIds)")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens to process")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help=f"Maximum parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Query batch size (default: {BATCH_SIZE})")
    parser.add_argument("--sequential", action="store_true", help="Process tokens sequentially (no parallelism)")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        logger.error("GRAPH_API_KEY not found in environment variables")
        return
    
    # Create output directory
    os.makedirs(TRADES_DIR, exist_ok=True)
    
    # Extract token IDs from CSV
    tokens, market_tokens = extract_unique_tokens(args.input, args.column)
    
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
    
    # Process tokens
    if args.sequential:
        # Process sequentially
        logger.info(f"Processing {len(tokens)} tokens sequentially")
        results = []
        for token_id in tqdm(tokens, desc="Processing tokens"):
            result = process_token(token_id, api_key, TRADES_DIR)
            results.append(result)
    else:
        # Process in parallel
        workers = min(args.max_workers, len(tokens))
        logger.info(f"Processing {len(tokens)} tokens in parallel with {workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_token, token_id, api_key, TRADES_DIR) for token_id in tokens]
            
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tokens"):
                results.append(future.result())
    
    # Summarize results
    successful = sum(1 for r in results if r.get('success', False))
    with_trades = sum(1 for r in results if r.get('trade_count', 0) > 0)
    total_trades = sum(r.get('trade_count', 0) for r in results)
    
    logger.info(f"Processing complete: {successful}/{len(results)} tokens successful")
    logger.info(f"Tokens with trades: {with_trades}/{len(results)}")
    logger.info(f"Total trades collected: {total_trades:,}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'tokens_processed': len(results),
        'tokens_successful': successful,
        'tokens_with_trades': with_trades,
        'total_trades': total_trades,
        'results': results
    }
    
    with open(os.path.join(DATA_DIR, "collection_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {DATA_DIR}")

if __name__ == "__main__":
    main()