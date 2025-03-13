#!/usr/bin/env python3
"""
Order Event Inspector

This script queries a single OrderFilledEvent from Polymarket's subgraph and
inspects its structure, with special focus on how timestamps are stored.
This will help determine the correct format for timestamp-based filtering.

Usage:
    python timestamp_test.py --token-id <token_id>
"""

import os
import sys
import json
from datetime import datetime
import argparse
from dotenv import load_dotenv
import requests
import pandas as pd

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Inspect OrderFilledEvent data structure")
    parser.add_argument("--token-id", required=True, help="Token ID to query")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        print("Error: GRAPH_API_KEY not found in environment variables")
        return
    
    print(f"Using token ID: {args.token_id}")
    
    # Subgraph URL
    subgraph_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    
    # Set up headers for requests
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # First verify token exists
    print("\nVerifying token exists...")
    market_data_query = {
        'query': f"""
        {{
          marketData(id: "{args.token_id}") {{
            id
            condition {{
              id
            }}
            outcomeIndex
            priceOrderbook
          }}
        }}
        """
    }
    
    try:
        response = requests.post(
            subgraph_url,
            headers=headers,
            json=market_data_query,
            timeout=30  # Increase timeout to 30 seconds
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'marketData' in data['data'] and data['data']['marketData']:
                market_data = data['data']['marketData']
                condition_id = market_data['condition']['id'] if market_data['condition'] else "Unknown"
                print(f"Token verified! Associated condition: {condition_id}")
                print(f"Price: {market_data.get('priceOrderbook')}")
            else:
                print(f"Error: Token {args.token_id} not found")
                return
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return
    except Exception as e:
        print(f"Error verifying token: {e}")
        return
    
    # Approach 1: Query a single OrderFilledEvent
    print("\nApproach 1: Querying a single OrderFilledEvent")
    order_query = {
        'query': f"""
        {{
          orderFilledEvents(
            first: 1
            orderBy: timestamp
            orderDirection: desc
            where: {{
              makerAssetId: "{args.token_id}"
            }}
          ) {{
            id
            timestamp
            maker
            taker
            makerAssetId
            takerAssetId
            makerAmountFilled
            takerAmountFilled
            fee
            transactionHash
          }}
        }}
        """
    }
    
    try:
        response = requests.post(
            subgraph_url,
            headers=headers,
            json=order_query,
            timeout=60  # Increase timeout to 60 seconds
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'orderFilledEvents' in data['data'] and data['data']['orderFilledEvents']:
                event = data['data']['orderFilledEvents'][0]
                print("\nOrderFilledEvent found!")
                print("\nOrderFilledEvent structure:")
                
                # Print each field with its value and data type
                for field, value in event.items():
                    print(f"  {field}:")
                    print(f"    Value: {value}")
                    print(f"    Type: {type(value).__name__}")
                    
                    # If it's a timestamp, show human-readable date
                    if field == 'timestamp':
                        try:
                            # Try different interpretations of timestamp
                            as_int = int(value)
                            print(f"    As datetime: {datetime.fromtimestamp(as_int)}")
                            print(f"    Raw numeric value: {as_int}")
                        except:
                            print("    Could not convert to datetime")
                
                # Try to calculate price from amounts
                maker_amount = float(event['makerAmountFilled'])
                taker_amount = float(event['takerAmountFilled'])
                if maker_amount > 0:
                    price = taker_amount / maker_amount
                    print(f"\nCalculated price: {price}")
            else:
                print("No OrderFilledEvent found for this token")
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error querying OrderFilledEvent: {e}")
    
    # Approach 2: Try EnrichedOrderFilled entity
    print("\nApproach 2: Checking EnrichedOrderFilled entity")
    enriched_query = {
        'query': """
        {
          enrichedOrderFilleds(
            first: 1
            orderBy: timestamp
            orderDirection: desc
          ) {
            id
            timestamp
            market
            price
            size
            side
          }
        }
        """
    }
    
    try:
        response = requests.post(
            subgraph_url,
            headers=headers,
            json=enriched_query,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'enrichedOrderFilleds' in data['data'] and data['data']['enrichedOrderFilleds']:
                enriched_data = data['data']['enrichedOrderFilleds'][0]
                print("\nEnrichedOrderFilled entity:")
                print(json.dumps(enriched_data, indent=2))
                
                # Examine timestamp
                if 'timestamp' in enriched_data:
                    timestamp = enriched_data['timestamp']
                    print(f"\nTimestamp field:")
                    print(f"  Raw value: {timestamp}")
                    print(f"  Type: {type(timestamp).__name__}")
                    
                    # Try to interpret as datetime
                    try:
                        ts_int = int(timestamp)
                        print(f"  As datetime: {datetime.fromtimestamp(ts_int)}")
                    except:
                        print("  Could not convert to datetime")
            else:
                print("No EnrichedOrderFilled entities found")
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error checking EnrichedOrderFilled: {e}")
    
    # Approach 3: Query orderbook entity
    print("\nApproach 3: Checking Orderbook entity")
    orderbook_query = {
        'query': f"""
        {{
          orderbook(id: "{args.token_id}") {{
            id
            tradesQuantity
            buysQuantity
            sellsQuantity
            collateralVolume
            scaledCollateralVolume
          }}
        }}
        """
    }
    
    try:
        response = requests.post(
            subgraph_url,
            headers=headers,
            json=orderbook_query,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'orderbook' in data['data'] and data['data']['orderbook']:
                orderbook_data = data['data']['orderbook']
                print("\nOrderbook entity:")
                print(json.dumps(orderbook_data, indent=2))
                print(f"\nTotal trades: {orderbook_data.get('tradesQuantity')}")
                print(f"Volume: {orderbook_data.get('scaledCollateralVolume')}")
            else:
                print("No Orderbook entity found")
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error checking Orderbook: {e}")
    
    print("\nInspection complete.")
    
    # Suggestions
    print("\nBased on common subgraph patterns, timestamps are likely stored as:")
    print("1. String representation of Unix timestamp (seconds since epoch)")
    print("2. When querying, timestamps should be passed as strings in GraphQL queries")
    print("3. For filtering, use 'timestamp_gte: \"1234567890\"' format")

if __name__ == "__main__":
    main()