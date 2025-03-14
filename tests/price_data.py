#!/usr/bin/env python3
"""
Stream-Processing Polymarket Metrics Calculator

Calculates key pre-election metrics for Polymarket tokens using stream processing:
- Closing price (last trade before election)
- Pre-election VWAP (48-hour window)
- Price volatility (48-hour window)

This version processes trades in batches, updating metrics incrementally to handle
large volumes efficiently.

Usage:
    python price_data.py --token-ids <token_ids_str> --election-date <YYYY-MM-DD> [--market-id <market_id>]
"""

import os
import sys
import json
import ast
import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor

# Constants
VWAP_WINDOW_HOURS = 48
MAX_BATCH_SIZE = 10000  
MAX_WORKERS = 2
OUTPUT_DIR = "prediction_metrics"

class StreamingStats:
    """Class to calculate streaming statistics without storing all data."""
    
    def __init__(self):
        # VWAP tracking
        self.total_value = 0.0
        self.total_volume = 0.0
        self.trade_count = 0
        
        # Buy/Sell tracking
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        
        # Volatility tracking (Welford's algorithm)
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from the mean
        self.min_price = float('inf')
        self.max_price = float('-inf')
        
        # Sample price tracking (for diagnostics/verification)
        self.prices_sample = []
        self.sample_rate = 0.001  # Store 0.1% of prices for verification
    
    def update(self, price, size, side):
        """Update all statistics with a new trade."""
        price = float(price)
        size = float(size)
        
        # Update VWAP components
        self.total_value += price * size
        self.total_volume += size
        self.trade_count += 1
        
        # Update Buy/Sell volumes
        if side == 'Buy':
            self.buy_volume += size
        elif side == 'Sell':
            self.sell_volume += size
        
        # Update volatility components (Welford's online algorithm)
        old_mean = self.mean
        self.mean += (price - old_mean) / self.trade_count
        self.m2 += (price - old_mean) * (price - self.mean)
        
        # Update min/max
        self.min_price = min(self.min_price, price)
        self.max_price = max(self.max_price, price)
        
        # Store occasional samples for verification
        if np.random.random() < self.sample_rate:
            self.prices_sample.append(price)
    
    def get_vwap(self):
        """Get the volume-weighted average price."""
        if self.total_volume > 0:
            return self.total_value / self.total_volume
        return None
    
    def get_buy_sell_ratio(self):
        """Get buy volume to sell volume ratio."""
        if self.sell_volume > 0:
            return self.buy_volume / self.sell_volume
        return None
    
    def get_volatility(self):
        """Get price volatility as coefficient of variation."""
        if self.trade_count < 2:
            return None
            
        # Calculate standard deviation from m2
        variance = self.m2 / (self.trade_count - 1)
        std_dev = np.sqrt(variance)
        
        # Return coefficient of variation
        if self.mean > 0:
            return std_dev / self.mean
        return None
    
    def get_stats(self):
        """Get all calculated statistics."""
        return {
            'vwap': self.get_vwap(),
            'trade_count': self.trade_count,
            'total_volume': float(self.total_volume),
            'buy_volume': float(self.buy_volume),
            'sell_volume': float(self.sell_volume),
            'buy_sell_ratio': self.get_buy_sell_ratio(),
            'price_volatility': self.get_volatility(),
            'mean_price': float(self.mean) if self.trade_count > 0 else None,
            'min_price': float(self.min_price) if self.min_price != float('inf') else None,
            'max_price': float(self.max_price) if self.max_price != float('-inf') else None,
            'price_range': float(self.max_price - self.min_price) 
                           if self.min_price != float('inf') and self.max_price != float('-inf') 
                           else None,
            'sample_count': len(self.prices_sample)
        }

def parse_token_ids(token_ids_str):
    """Parse token IDs from string to list."""
    try:
        if token_ids_str.startswith('[') and token_ids_str.endswith(']'):
            if '"' in token_ids_str or "'" in token_ids_str:
                try:
                    return json.loads(token_ids_str)
                except:
                    return ast.literal_eval(token_ids_str)
            else:
                return token_ids_str.strip('[]').split(',')
        else:
            return token_ids_str.split(',')
    except Exception as e:
        print(f"Error parsing token IDs: {e}")
        return []

def get_closing_price(token_id, election_timestamp, api_key):
    """Get the last trade before election."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    query = {
        'query': f"""
        {{
          enrichedOrderFilleds(
            first: 1
            orderBy: timestamp
            orderDirection: desc
            where: {{
              market: "{token_id}",
              timestamp_lt: "{election_timestamp}"
            }}
          ) {{
            id
            timestamp
            price
            side
            size
          }}
        }}
        """
    }
    
    try:
        response = requests.post(url, headers=headers, json=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            trades = data.get('data', {}).get('enrichedOrderFilleds', [])
            
            if trades:
                trade = trades[0]
                trade_time = datetime.fromtimestamp(int(trade['timestamp']))
                hours_before = (election_timestamp - int(trade['timestamp'])) / 3600
                
                print(f"Found last trade before election!")
                print(f"  Time: {trade_time} ({hours_before:.1f} hours before election)")
                print(f"  Price: {trade['price']}")
                
                return {
                    'closing_price': float(trade['price']),
                    'closing_time': trade_time.isoformat(),
                    'hours_before_election': hours_before
                }
    except Exception as e:
        print(f"Error fetching closing price: {e}")
    
    return {'closing_price': None}

def process_trades_streaming(token_id, start_timestamp, end_timestamp, api_key, progress_interval=10000):
    """Process trades in batches using streaming statistics."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    # Initialize streaming stats calculator
    stats = StreamingStats()
    
    skip = 0
    has_more = True
    last_progress = 0
    
    print(f"Streaming trades for 48-hour window before election...")
    start_time = time.time()
    
    while has_more:
        query = {
            'query': f"""
            {{
              enrichedOrderFilleds(
                first: {MAX_BATCH_SIZE}
                skip: {skip}
                orderBy: timestamp
                orderDirection: desc
                where: {{
                  market: "{token_id}",
                  timestamp_gte: "{start_timestamp}",
                  timestamp_lt: "{end_timestamp}"
                }}
              ) {{
                price
                size
                side
              }}
            }}
            """
        }
        
        try:
            response = requests.post(url, headers=headers, json=query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                trades = data.get('data', {}).get('enrichedOrderFilleds', [])
                
                if not trades:
                    has_more = False
                else:
                    # Update streaming stats with each trade
                    for trade in trades:
                        stats.update(trade['price'], trade['size'], trade['side'])
                    
                    # Update counters
                    batch_size = len(trades)
                    skip += batch_size
                    
                    # Show progress at regular intervals
                    if skip >= last_progress + progress_interval:
                        elapsed = time.time() - start_time
                        rate = skip / elapsed if elapsed > 0 else 0
                        print(f"  Processed {skip} trades ({rate:.1f} trades/sec)... Current VWAP: {stats.get_vwap():.4f}")
                        last_progress = skip
                    
                    # Check if we've reached the end
                    if batch_size < MAX_BATCH_SIZE:
                        has_more = False
            else:
                print(f"Error: Status code {response.status_code}")
                has_more = False
        except Exception as e:
            print(f"Error processing trades: {e}")
            has_more = False
    
    # Calculate final runtime
    total_time = time.time() - start_time
    trades_per_sec = stats.trade_count / total_time if total_time > 0 else 0
    
    print(f"Completed processing {stats.trade_count} trades in {total_time:.2f} seconds ({trades_per_sec:.1f} trades/sec)")
    return stats.get_stats()

def process_token(token_id, election_timestamp, api_key):
    """Process a token to get all metrics."""
    print(f"\n{'='*50}")
    print(f"Processing token: {token_id}")
    print(f"{'='*50}")
    
    result = {'token_id': token_id}
    
    # 1. Get closing price (last trade before election)
    closing_data = get_closing_price(token_id, election_timestamp, api_key)
    result.update(closing_data)
    
    # 2. Stream process trades for the 48-hour window before election
    window_start = election_timestamp - (VWAP_WINDOW_HOURS * 3600)
    metrics = process_trades_streaming(token_id, window_start, election_timestamp, api_key)
    result.update(metrics)
    
    # Print key metrics
    print("\nFinal metrics:")
    if closing_data.get('closing_price'):
        print(f"  Closing price: {closing_data['closing_price']} ({closing_data.get('hours_before_election', 'N/A')} hours before election)")
    if metrics.get('vwap'):
        print(f"  VWAP: {metrics['vwap']:.4f} (from {metrics['trade_count']:,} trades)")
    if metrics.get('price_volatility'):
        print(f"  Volatility: {metrics['price_volatility']:.6f}")
    if metrics.get('buy_sell_ratio'):
        print(f"  Buy/Sell ratio: {metrics['buy_sell_ratio']:.4f}")
    if metrics.get('price_range'):
        print(f"  Price range: {metrics['price_range']:.4f} ({metrics['min_price']:.4f} - {metrics['max_price']:.4f})")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Stream-processing metrics calculator for Polymarket tokens")
    parser.add_argument("--token-ids", required=True, help="Token IDs as JSON string")
    parser.add_argument("--election-date", required=True, help="Election date (YYYY-MM-DD)")
    parser.add_argument("--market-id", help="Market ID (optional)")
    parser.add_argument("--output", default="price_metrics", help="Output filename")
    
    args = parser.parse_args()
    start_time = time.time()
    
    # Load API key
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    if not api_key:
        print("Error: GRAPH_API_KEY not found in environment variables")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Parse inputs
    token_ids = parse_token_ids(args.token_ids)
    if not token_ids:
        print("Error: No valid token IDs found")
        return
    
    print(f"Parsed {len(token_ids)} token IDs")
    
    # Parse election date
    try:
        election_date = datetime.fromisoformat(args.election_date.replace('Z', '+00:00')) if 'T' in args.election_date else datetime.strptime(args.election_date, '%Y-%m-%d') 
        election_date = election_date.replace(hour=0, minute=0, second=0, microsecond=0)
        election_timestamp = int(election_date.timestamp())
        print(f"Election date: {election_date} (timestamp: {election_timestamp})")
    except Exception as e:
        print(f"Error parsing election date: {e}")
        return
    
    # Process tokens sequentially (streaming already uses multiple threads internally)
    results = {
        'market_id': args.market_id,
        'election_date': args.election_date,
        'election_timestamp': election_timestamp,
        'analysis_timestamp': datetime.now().isoformat(),
        'tokens': []
    }
    
    for token_id in token_ids:
        try:
            token_result = process_token(token_id, election_timestamp, api_key)
            results['tokens'].append(token_result)
        except Exception as e:
            print(f"Error processing token {token_id}: {e}")
            results['tokens'].append({'token_id': token_id, 'error': str(e)})
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"{args.output}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    # Print summary
    print("\nSummary of results:")
    print(f"{'='*50}")
    for token in results['tokens']:
        token_id_short = token['token_id'][-8:] if len(token['token_id']) > 8 else token['token_id']
        if token.get('closing_price') is not None:
            print(f"Token {token_id_short}: Closing price = {token['closing_price']} ({token.get('hours_before_election', 0):.1f} hours before election)")
        else:
            print(f"Token {token_id_short}: No trades found")

if __name__ == "__main__":
    main()