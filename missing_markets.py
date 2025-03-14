#!/usr/bin/env python3
"""
Generate Missing Markets Report

Creates a CSV report of markets that have missing token data by matching
missing tokens with their original market information from the CSV.
"""

import os
import json
import pandas as pd
import argparse
from datetime import datetime

def parse_token_ids(token_ids_str):
    """Parse token IDs from string representation in CSV."""
    try:
        if not token_ids_str or pd.isna(token_ids_str):
            return []
            
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

def generate_missing_markets_report(csv_file, missing_tokens_file, output_file):
    """Generate a report of markets with missing token data."""
    # Load the missing tokens
    with open(missing_tokens_file, 'r') as f:
        report = json.load(f)
    
    missing_token_ids = set(report.get('missing_token_ids', []))
    
    if not missing_token_ids:
        print("No missing tokens found in the report")
        return
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize lists to store market data
    market_data = []
    
    # Check each row in the CSV for missing tokens
    for _, row in df.iterrows():
        try:
            token_ids = parse_token_ids(row.get('clobTokenIds', ''))
            
            # Check if any of these tokens are in the missing tokens list
            missing = [token for token in token_ids if token in missing_token_ids]
            
            if missing:
                # Extract relevant market info
                market_info = {
                    'question': row.get('question', 'Unknown'),
                    'missing_tokens': missing,
                    'missing_token_count': len(missing),
                    'total_token_count': len(token_ids),
                    'volume': row.get('volume', 0),
                    'volume_num': row.get('volumeNum', 0),
                    'volume_clob': row.get('volumeClob', 0),
                    'liquidity': row.get('liquidity', 0),
                    'active': row.get('active', False),
                    'closed': row.get('closed', False),
                    'restricted': row.get('restricted', False),
                    'start_date': row.get('startDate', None),
                    'end_date': row.get('endDate', None),
                    'enable_order_book': row.get('enableOrderBook', None),
                    'market_maker_address': row.get('marketMakerAddress', None),
                    'slug': row.get('slug', None),
                    'condition_id': row.get('conditionId', None),
                    'event_title': row.get('event_title', None),
                    'event_description': row.get('event_description', None),
                    'event_country': row.get('event_countryName', None),
                    'event_election_type': row.get('event_electionType', None)
                }
                market_data.append(market_info)
        except Exception as e:
            print(f"Error processing row: {e}")
    
    if not market_data:
        print("No markets found with missing tokens")
        return
    
    # Convert to DataFrame for easier manipulation
    markets_df = pd.DataFrame(market_data)
    
    # Calculate some statistics
    total_markets = len(markets_df)
    total_missing_tokens = sum(markets_df['missing_token_count'])
    restricted_markets = sum(markets_df['restricted'] == True)
    active_markets = sum(markets_df['active'] == True)
    closed_markets = sum(markets_df['closed'] == True)
    orderbook_enabled = sum(markets_df['enable_order_book'] == True)
    
    # Print summary
    print(f"Found {total_markets} markets with {total_missing_tokens} missing tokens")
    print(f"Restricted markets: {restricted_markets}")
    print(f"Active markets: {active_markets}")
    print(f"Closed markets: {closed_markets}")
    print(f"Markets with orderbook enabled: {orderbook_enabled}")
    
    # Save full data to JSON
    json_output = output_file.replace('.csv', '.json')
    with open(json_output, 'w') as f:
        json.dump(market_data, f, indent=2)
    
    # Simplified CSV output
    csv_columns = [
        'question', 'missing_token_count', 'total_token_count', 
        'volume', 'restricted', 'active', 'closed', 'enable_order_book',
        'event_country', 'event_election_type', 'start_date', 'end_date'
    ]
    
    # Convert missing_tokens list to string for CSV
    markets_df['missing_tokens'] = markets_df['missing_tokens'].apply(lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else ''))
    
    # Save to CSV
    markets_df.to_csv(output_file, index=False, columns=csv_columns)
    
    print(f"Full report saved to {json_output}")
    print(f"Summary CSV saved to {output_file}")
    
    # Generate insights
    print("\nINSIGHTS:")
    
    # Check if there are patterns in restrictions
    if restricted_markets > 0:
        pct_restricted = (restricted_markets / total_markets) * 100
        print(f"- {pct_restricted:.1f}% of missing markets are marked as restricted")
    
    # Check patterns in countries
    countries = markets_df['event_country'].value_counts()
    if not countries.empty:
        print("- Top countries with missing data:")
        for country, count in countries.head(3).items():
            if not pd.isna(country):
                print(f"  * {country}: {count} markets")
    
    # Check patterns in election types
    election_types = markets_df['event_election_type'].value_counts()
    if not election_types.empty:
        print("- Top election types with missing data:")
        for etype, count in election_types.head(3).items():
            if not pd.isna(etype):
                print(f"  * {etype}: {count} markets")
    
    # Check time patterns
    try:
        markets_df['start_date'] = pd.to_datetime(markets_df['start_date'], errors='coerce')
        markets_df['end_date'] = pd.to_datetime(markets_df['end_date'], errors='coerce')
        
        if not markets_df['start_date'].isna().all():
            earliest = markets_df['start_date'].min()
            latest = markets_df['start_date'].max()
            print(f"- Time range of missing markets: {earliest.date()} to {latest.date()}")
            
            # Check for clustering in time periods
            markets_df['start_year'] = markets_df['start_date'].dt.year
            year_counts = markets_df['start_year'].value_counts()
            
            if not year_counts.empty:
                print("- Distribution by year:")
                for year, count in year_counts.items():
                    print(f"  * {year}: {count} markets")
    except Exception as e:
        print(f"Error analyzing dates: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Missing Markets Report")
    parser.add_argument("--csv", required=True, help="Path to original CSV file")
    parser.add_argument("--missing-tokens", default="reports/missing_tokens.json", help="Missing tokens report")
    parser.add_argument("--output", default="reports/missing_markets.csv", help="Output CSV file")
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        return
    
    if not os.path.exists(args.missing_tokens):
        print(f"ERROR: Missing tokens file not found: {args.missing_tokens}")
        return
    
    # Make sure reports directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate the report
    generate_missing_markets_report(args.csv, args.missing_tokens, args.output)

if __name__ == "__main__":
    main()