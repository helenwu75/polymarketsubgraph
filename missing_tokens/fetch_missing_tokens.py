#!/usr/bin/env python3
"""
Fetch Missing Tokens Script

Fetches trade data only for tokens that weren't successfully captured in the initial run.
"""

import os
import json
import time
import argparse
import requests
from tqdm import tqdm
import pandas as pd
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("missing_tokens_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 1000
MAX_RETRIES = 5
RETRY_DELAY = 10
DATA_DIR = "polymarket_raw_data"
TRADES_DIR = f"{DATA_DIR}/trades"

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
                    response = requests.post(url, headers=headers, json=query, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        trades = data.get('data', {}).get('enrichedOrderFilleds', [])
                        
                        # Process the trades
                        if trades:
                            # Add trades to our collection
                            all_trades.extend(trades)
                            
                            # Move to next batch
                            skip += len(trades)
                            
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
                            time.sleep(RETRY_DELAY * (2 ** retry))  # Exponential backoff
                        else:
                            raise Exception(f"Failed after {MAX_RETRIES} retries")
                
                except Exception as e:
                    logger.warning(f"Error on retry {retry+1}/{MAX_RETRIES}: {e}")
                    if retry < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (2 ** retry))  # Exponential backoff
                    else:
                        raise
        
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
            
            # Save to parquet file
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

def main():
    parser = argparse.ArgumentParser(description="Fetch Missing Tokens Data")
    parser.add_argument("--report", default="reports/missing_tokens.json", help="Missing tokens report file")
    parser.add_argument("--output-dir", default=TRADES_DIR, help="Output directory for trade data")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        logger.error("GRAPH_API_KEY not found in environment variables")
        return
    
    # Load missing tokens report
    if not os.path.exists(args.report):
        logger.error(f"Report file not found: {args.report}")
        return
    
    with open(args.report, 'r') as f:
        report = json.load(f)
    
    missing_tokens = report.get('missing_token_ids', [])
    
    if not missing_tokens:
        logger.info("No missing tokens found in report")
        return
    
    logger.info(f"Found {len(missing_tokens)} missing tokens to fetch")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each missing token
    results = []
    for token_id in tqdm(missing_tokens, desc="Fetching missing tokens"):
        result = collect_token_trades(token_id, api_key, args.output_dir)
        results.append(result)
        # Add a small delay between requests
        time.sleep(1)
    
    # Summarize results
    successful = sum(1 for r in results if r.get('success', False))
    with_trades = sum(1 for r in results if r.get('trade_count', 0) > 0)
    total_trades = sum(r.get('trade_count', 0) for r in results)
    
    logger.info(f"Processing complete: {successful}/{len(results)} tokens successful")
    logger.info(f"Tokens with trades: {with_trades}/{len(results)}")
    logger.info(f"Total trades collected: {total_trades:,}")
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tokens_processed': len(results),
        'tokens_successful': successful,
        'tokens_with_trades': with_trades,
        'total_trades': total_trades,
        'results': results
    }
    
    with open(os.path.join("reports", "missing_tokens_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to reports/missing_tokens_results.json")

if __name__ == "__main__":
    main()