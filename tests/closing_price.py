#!/usr/bin/env python3
"""
Polymarket Election Closing Price Finder

This script retrieves the closing price (last price before the election date) for 
Polymarket election markets using the EnrichedOrderFilled entity, which has been 
verified to work with the subgraph.

Usage:
    python closing_price_finder.py --token-ids <token_ids_str> 
                                  --election-date <YYYY-MM-DD>
                                  [--market-id <market_id>]
                                  [--days-before <days>]
                                  [--output <filename>]
"""

import os
import sys
import json
import ast
from datetime import datetime
import argparse
import pandas as pd
from dotenv import load_dotenv
import requests

# Configure constants
DEFAULT_PRE_ELECTION_DAYS = 7  # Days before election to look for trades
OUTPUT_DIRECTORY = "prediction_metrics"  # Directory to store results

def parse_token_ids(token_ids_str):
    """Parse token IDs from a string representation."""
    if not token_ids_str or token_ids_str == 'nan' or token_ids_str == 'None':
        return []
        
    try:
        # Try to parse the token IDs string
        if token_ids_str.startswith('[') and token_ids_str.endswith(']'):
            if '"' in token_ids_str or "'" in token_ids_str:
                try:
                    token_ids = json.loads(token_ids_str)
                except json.JSONDecodeError:
                    token_ids = ast.literal_eval(token_ids_str)
            else:
                token_ids = token_ids_str.strip('[]').split(',')
        else:
            token_ids = token_ids_str.split(',')
        
        # Clean up whitespace
        token_ids = [id.strip().strip('"\'') for id in token_ids]
        return token_ids
    except Exception as e:
        print(f"Error parsing token IDs string: {e}")
        return [token_ids_str]  # Return as a single ID

def parse_election_date(date_str):
    """Parse election date string into datetime object."""
    try:
        # Try parsing ISO format first (if time is included)
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            # If that fails, try simple YYYY-MM-DD format
            return datetime.strptime(date_str, '%Y-%m-%d')
    except Exception as e:
        raise ValueError(f"Invalid election date format. Please use YYYY-MM-DD. Error: {e}")

def get_token_info(token_id, api_key, subgraph_url):
    """
    Get basic token information from the subgraph.
    
    Args:
        token_id (str): Token ID to query
        api_key (str): Graph API key
        subgraph_url (str): Subgraph URL
        
    Returns:
        dict: Token information
    """
    # Set up headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Query for token info
    query = {
        'query': f"""
        {{
          marketData(id: "{token_id}") {{
            id
            condition {{
              id
            }}
            outcomeIndex
            priceOrderbook
          }}
          orderbook(id: "{token_id}") {{
            id
            tradesQuantity
            buysQuantity
            sellsQuantity
            scaledCollateralVolume
          }}
        }}
        """
    }
    
    try:
        response = requests.post(
            subgraph_url,
            headers=headers,
            json=query,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract market data
            token_info = {
                'token_id': token_id,
                'found': False,
                'current_price': None,
                'condition_id': None,
                'outcome_index': None,
                'trades_quantity': 0,
                'volume': 0
            }
            
            # Check for market data
            if data.get('data', {}).get('marketData'):
                market_data = data['data']['marketData']
                token_info['found'] = True
                token_info['current_price'] = market_data.get('priceOrderbook')
                token_info['condition_id'] = market_data.get('condition', {}).get('id')
                token_info['outcome_index'] = market_data.get('outcomeIndex')
            
            # Check for orderbook data
            if data.get('data', {}).get('orderbook'):
                orderbook = data['data']['orderbook']
                token_info['trades_quantity'] = orderbook.get('tradesQuantity')
                token_info['buys_quantity'] = orderbook.get('buysQuantity')
                token_info['sells_quantity'] = orderbook.get('sellsQuantity')
                token_info['volume'] = orderbook.get('scaledCollateralVolume')
            
            return token_info
        else:
            print(f"Error getting token info: Status {response.status_code}")
            print(f"Response: {response.text}")
            return {
                'token_id': token_id,
                'found': False,
                'error': f"Status {response.status_code}"
            }
    except Exception as e:
        print(f"Error getting token info: {e}")
        return {
            'token_id': token_id,
            'found': False,
            'error': str(e)
        }

def get_closing_price(token_id, election_timestamp, days_before, api_key, subgraph_url):
    """
    Get the closing price (last trade before election) for a token.
    
    Args:
        token_id (str): Token ID to query
        election_timestamp (int): Unix timestamp of election date
        days_before (int): Number of days before election to look
        api_key (str): Graph API key
        subgraph_url (str): Subgraph URL
        
    Returns:
        dict: Closing price information
    """
    # Calculate start timestamp
    start_timestamp = election_timestamp - (days_before * 24 * 60 * 60)
    
    print(f"Searching for trades between:")
    print(f"  Start: {datetime.fromtimestamp(start_timestamp)} ({start_timestamp})")
    print(f"  End: {datetime.fromtimestamp(election_timestamp)} ({election_timestamp})")
    
    # Set up headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Query for last trade before election
    query = {
        'query': f"""
        {{
          enrichedOrderFilleds(
            first: 1
            orderBy: timestamp
            orderDirection: desc
            where: {{
              market: "{token_id}",
              timestamp_lt: "{election_timestamp}",
              timestamp_gte: "{start_timestamp}"
            }}
          ) {{
            id
            timestamp
            price
            side
            size
            transactionHash
          }}
        }}
        """
    }
    
    try:
        response = requests.post(
            subgraph_url,
            headers=headers,
            json=query,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('data', {}).get('enrichedOrderFilleds') and len(data['data']['enrichedOrderFilleds']) > 0:
                trade = data['data']['enrichedOrderFilleds'][0]
                trade_time = datetime.fromtimestamp(int(trade['timestamp']))
                
                hours_before_election = (election_timestamp - int(trade['timestamp'])) / 3600
                
                print(f"Found last trade before election!")
                print(f"  Time: {trade_time} ({hours_before_election:.1f} hours before election)")
                print(f"  Price: {trade['price']}")
                print(f"  Side: {trade['side']}")
                print(f"  Size: {trade['size']}")
                
                return {
                    'found': True,
                    'timestamp': int(trade['timestamp']),
                    'datetime': trade_time.isoformat(),
                    'price': float(trade['price']),
                    'side': trade['side'],
                    'size': float(trade['size']),
                    'hours_before_election': hours_before_election,
                    'transaction_hash': trade.get('transactionHash')
                }
            else:
                print("No trades found in the specified time period")
                
                # If no trades found in the specified period, try extending the search
                extended_start = start_timestamp - (30 * 24 * 60 * 60)  # Go back another 30 days
                print(f"\nExtending search to {datetime.fromtimestamp(extended_start)}")
                
                extended_query = {
                    'query': f"""
                    {{
                      enrichedOrderFilleds(
                        first: 1
                        orderBy: timestamp
                        orderDirection: desc
                        where: {{
                          market: "{token_id}",
                          timestamp_lt: "{election_timestamp}",
                          timestamp_gte: "{extended_start}"
                        }}
                      ) {{
                        id
                        timestamp
                        price
                        side
                        size
                        transactionHash
                      }}
                    }}
                    """
                }
                
                extended_response = requests.post(
                    subgraph_url,
                    headers=headers,
                    json=extended_query,
                    timeout=60
                )
                
                if extended_response.status_code == 200:
                    extended_data = extended_response.json()
                    
                    if extended_data.get('data', {}).get('enrichedOrderFilleds') and len(extended_data['data']['enrichedOrderFilleds']) > 0:
                        extended_trade = extended_data['data']['enrichedOrderFilleds'][0]
                        extended_time = datetime.fromtimestamp(int(extended_trade['timestamp']))
                        
                        extended_hours = (election_timestamp - int(extended_trade['timestamp'])) / 3600
                        
                        print(f"Found a trade with extended search!")
                        print(f"  Time: {extended_time} ({extended_hours:.1f} hours before election)")
                        print(f"  Price: {extended_trade['price']}")
                        
                        return {
                            'found': True,
                            'timestamp': int(extended_trade['timestamp']),
                            'datetime': extended_time.isoformat(),
                            'price': float(extended_trade['price']),
                            'side': extended_trade['side'],
                            'size': float(extended_trade['size']),
                            'hours_before_election': extended_hours,
                            'transaction_hash': extended_trade.get('transactionHash'),
                            'extended_search': True
                        }
                    else:
                        print("No trades found even with extended search")
                
                return {
                    'found': False,
                    'error': "No trades found in the specified time period"
                }
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return {
                'found': False,
                'error': f"API error: Status {response.status_code}"
            }
    except Exception as e:
        print(f"Error getting closing price: {e}")
        return {
            'found': False,
            'error': str(e)
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Find closing prices for Polymarket election markets")
    parser.add_argument("--token-ids", required=True, help="Token IDs as JSON string")
    parser.add_argument("--election-date", required=True, help="Election date (YYYY-MM-DD)")
    parser.add_argument("--market-id", help="Market ID (optional)")
    parser.add_argument("--days-before", type=int, default=DEFAULT_PRE_ELECTION_DAYS, 
                        help=f"Days before election to search (default: {DEFAULT_PRE_ELECTION_DAYS})")
    parser.add_argument("--output", help="Output filename", default="closing_prices")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        print("Error: GRAPH_API_KEY not found in environment variables")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Process token IDs
    token_ids = parse_token_ids(args.token_ids)
    if not token_ids:
        print("Error: No valid token IDs found")
        return
    
    print(f"Parsed {len(token_ids)} token IDs")
    
    # Parse election date
    try:
        election_date = parse_election_date(args.election_date)
        # Set to midnight of that day
        election_date = election_date.replace(hour=0, minute=0, second=0, microsecond=0)
        election_timestamp = int(election_date.timestamp())
        
        print(f"Election date: {election_date} (timestamp: {election_timestamp})")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Subgraph URL
    subgraph_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    
    # Initialize results
    results = {
        'market_id': args.market_id,
        'election_date': args.election_date,
        'election_timestamp': election_timestamp,
        'analysis_timestamp': datetime.now().isoformat(),
        'tokens': []
    }
    
    # Process each token
    for token_id in token_ids:
        print(f"\n{'='*50}")
        print(f"Processing token: {token_id}")
        print(f"{'='*50}")
        
        # Get token information
        token_info = get_token_info(token_id, api_key, subgraph_url)
        
        if not token_info.get('found', False):
            print(f"Token {token_id} not found or could not retrieve information")
            results['tokens'].append({
                'token_id': token_id,
                'found': False,
                'error': token_info.get('error', "Token not found")
            })
            continue
        
        print(f"Token found!")
        print(f"  Current price: {token_info.get('current_price')}")
        print(f"  Total trades: {token_info.get('trades_quantity')}")
        print(f"  Volume: {token_info.get('volume')}")
        
        # Get closing price
        closing_price_info = get_closing_price(token_id, election_timestamp, args.days_before, api_key, subgraph_url)
        
        # Combine token info with closing price info
        token_result = {**token_info, **closing_price_info}
        results['tokens'].append(token_result)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIRECTORY, f"{args.output}_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\nSummary of results:")
    print(f"{'='*50}")
    
    for token in results['tokens']:
        token_id_short = token['token_id'][-8:] if len(token['token_id']) > 8 else token['token_id']
        
        if token.get('found', False):
            if closing_price_info.get('found', False):
                price = token.get('price')
                time_before = token.get('hours_before_election', 0)
                print(f"Token {token_id_short}: Closing price = {price} ({time_before:.1f} hours before election)")
            else:
                current_price = token.get('current_price')
                print(f"Token {token_id_short}: No trades found. Current price = {current_price}")
        else:
            print(f"Token {token_id_short}: Not found or error occurred")

if __name__ == "__main__":
    main()