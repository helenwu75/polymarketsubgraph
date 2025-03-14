#!/usr/bin/env python3
"""
Find Missing Tokens Script

Identifies tokens that weren't successfully captured by comparing
the original CSV with collected data files.
"""

import os
import json
import pandas as pd
import glob
from typing import List, Dict, Set
import argparse

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
                    try:
                        import ast
                        return ast.literal_eval(token_ids_str)
                    except:
                        return []
            else:
                return token_ids_str.split(',')
        return []
    except Exception as e:
        print(f"Error parsing token IDs: {e}")
        return []

def get_csv_tokens(csv_file: str, token_col: str = 'clobTokenIds') -> Set[str]:
    """Extract all token IDs from the CSV file."""
    df = pd.read_csv(csv_file)
    
    if token_col not in df.columns:
        raise ValueError(f"Column '{token_col}' not found in CSV")
    
    all_tokens = set()
    for _, row in df.iterrows():
        token_ids = parse_token_ids(row[token_col])
        all_tokens.update(token_ids)
        
    # Remove any empty strings
    return {t for t in all_tokens if t}

def get_collected_tokens(trades_dir: str) -> Set[str]:
    """Get tokens that have been successfully collected."""
    parquet_files = glob.glob(os.path.join(trades_dir, "*.parquet"))
    return {os.path.splitext(os.path.basename(f))[0] for f in parquet_files}

def find_missing_tokens(csv_file: str, trades_dir: str, token_col: str = 'clobTokenIds') -> Dict:
    """Find tokens that exist in CSV but don't have collected data."""
    csv_tokens = get_csv_tokens(csv_file, token_col)
    collected_tokens = get_collected_tokens(trades_dir)
    
    missing_tokens = csv_tokens - collected_tokens
    empty_tokens = {t for t in collected_tokens if os.path.getsize(os.path.join(trades_dir, f"{t}.parquet")) < 1000}
    
    return {
        'total_csv_tokens': len(csv_tokens),
        'collected_tokens': len(collected_tokens),
        'missing_tokens': len(missing_tokens),
        'empty_tokens': len(empty_tokens),
        'missing_token_ids': sorted(list(missing_tokens)),
        'empty_token_ids': sorted(list(empty_tokens))
    }

def analyze_log_file(log_file: str) -> Dict:
    """Analyze log file to find tokens with no trades."""
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
    
    no_trades_tokens = set()
    for i, line in enumerate(log_lines):
        if "Collecting trades for token" in line:
            token_id = line.split("token ")[-1].strip()
            # Check if next line indicates no trades
            if i+1 < len(log_lines) and "No trades found for token" in log_lines[i+1]:
                no_trades_tokens.add(token_id)
    
    return {
        'tokens_with_no_trades': len(no_trades_tokens),
        'no_trades_token_ids': sorted(list(no_trades_tokens))
    }

def main():
    parser = argparse.ArgumentParser(description="Find missing tokens")
    parser.add_argument("--csv", required=True, default="top_election_markets.csv", help="Path to original CSV file")
    parser.add_argument("--trades-dir", default="polymarket_raw_data/trades", help="Directory with trade data")
    parser.add_argument("--log-file", default="data_collection.log", help="Log file path")
    parser.add_argument("--output", default="missing_tokens.json", help="Output JSON file")
    parser.add_argument("--token-col", default="clobTokenIds", help="Column with token IDs")
    args = parser.parse_args()
    
    # Make sure directories exist
    if not os.path.exists(args.csv):
        print(f"CSV file doesn't exist: {args.csv}")
        return
    
    if not os.path.exists(args.trades_dir):
        print(f"Trades directory doesn't exist: {args.trades_dir}")
        return
    
    # Find missing tokens
    results = find_missing_tokens(args.csv, args.trades_dir, args.token_col)
    
    # Add log analysis if log file exists
    if os.path.exists(args.log_file):
        log_results = analyze_log_file(args.log_file)
        results.update(log_results)
    
    # Print summary
    print(f"CSV tokens: {results['total_csv_tokens']}")
    print(f"Collected tokens: {results['collected_tokens']}")
    print(f"Missing tokens: {results['missing_tokens']}")
    
    if 'tokens_with_no_trades' in results:
        print(f"Tokens with no trades: {results['tokens_with_no_trades']}")
    
    if results['missing_tokens'] > 0:
        print("\nSample missing tokens:")
        for token in results['missing_token_ids'][:5]:
            print(f"  - {token}")
    
    # Save results
    
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.join(reports_dir, args.output)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to {output_path}")
    
    print(f"\nFull results saved to {args.output}")

if __name__ == "__main__":
    main()