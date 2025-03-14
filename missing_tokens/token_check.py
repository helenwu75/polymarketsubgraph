#!/usr/bin/env python3
"""
Polymarket Single Token Checker

A diagnostic script to investigate why a specific token isn't returning trade data.
This script tries multiple query approaches and prints detailed diagnostics.
"""

import os
import json
import requests
import time
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv('GRAPH_API_KEY')

if not api_key:
    print("Error: GRAPH_API_KEY not found in environment variables")
    exit(1)

# Target token ID
TOKEN_ID = "21351363155933934664313200388882515678031543240463621784219543817220104614446"

# Subgraph URLs
ORDERBOOK_URL = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
ACTIVITY_URL = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"

# Headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

def check_token_info():
    """Get basic token information."""
    print(f"\n=== Checking basic token info for {TOKEN_ID} ===")
    
    query = {
        'query': f"""
        {{
          marketData(id: "{TOKEN_ID}") {{
            id
            outcomeIndex
            priceOrderbook
            condition {{
              id
            }}
            fpmm {{
              id
              collateralVolume
              scaledCollateralVolume
            }}
          }}
        }}
        """
    }
    
    response = requests.post(ORDERBOOK_URL, headers=headers, json=query, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if 'data' in data and data['data'].get('marketData'):
            market_data = data['data']['marketData']
            print(f"\nToken exists with:")
            print(f"  Outcome Index: {market_data.get('outcomeIndex')}")
            print(f"  Current Price: {market_data.get('priceOrderbook')}")
            print(f"  Condition ID: {market_data.get('condition', {}).get('id')}")
            
            if market_data.get('fpmm'):
                print(f"  FPMM Address: {market_data['fpmm'].get('id')}")
                print(f"  Raw Volume: {market_data['fpmm'].get('collateralVolume')}")
                print(f"  Scaled Volume: {market_data['fpmm'].get('scaledCollateralVolume')}")
        else:
            print("Token does not exist in marketData entity")
    else:
        print(f"Request failed with status {response.status_code}: {response.text}")

def check_orderbook_data():
    """Check orderbook data for the token."""
    print(f"\n=== Checking orderbook data for {TOKEN_ID} ===")
    
    query = {
        'query': f"""
        {{
          orderbook(id: "{TOKEN_ID}") {{
            id
            tradesQuantity
            buysQuantity
            sellsQuantity
            collateralVolume
            scaledCollateralVolume
            scaledCollateralBuyVolume
            scaledCollateralSellVolume
            lastActiveDay
          }}
        }}
        """
    }
    
    response = requests.post(ORDERBOOK_URL, headers=headers, json=query, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if 'data' in data and data['data'].get('orderbook'):
            orderbook = data['data']['orderbook']
            print(f"\nOrderbook exists with:")
            print(f"  Trades Quantity: {orderbook.get('tradesQuantity')}")
            print(f"  Buys Quantity: {orderbook.get('buysQuantity')}")
            print(f"  Sells Quantity: {orderbook.get('sellsQuantity')}")
            print(f"  Collateral Volume: {orderbook.get('collateralVolume')}")
            print(f"  Scaled Volume: {orderbook.get('scaledCollateralVolume')}")
            
            if orderbook.get('lastActiveDay'):
                timestamp = int(orderbook['lastActiveDay'])
                last_active = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  Last Active: {last_active}")
        else:
            print("Token does not exist in orderbook entity")
    else:
        print(f"Request failed with status {response.status_code}: {response.text}")

def check_trade_history(strategy="enriched"):
    """Check trade history for the token using different query approaches."""
    print(f"\n=== Checking trade history using {strategy} strategy ===")
    
    if strategy == "enriched":
        # Try using enrichedOrderFilleds
        query = {
            'query': f"""
            {{
              enrichedOrderFilleds(
                first: 10
                orderBy: timestamp
                orderDirection: desc
                where: {{ market: "{TOKEN_ID}" }}
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
    elif strategy == "orderFilled":
        # Try using orderFilledEvents
        query = {
            'query': f"""
            {{
              orderFilledEvents(
                first: 10
                orderBy: timestamp
                orderDirection: desc
                where: {{ 
                  makerAssetId: "{TOKEN_ID}" 
                }}
              ) {{
                id
                timestamp
                makerAmountFilled
                takerAmountFilled
                transactionHash
              }}
            }}
            """
        }
    elif strategy == "market_position":
        # Try checking if there are any positions for this token
        query = {
            'query': f"""
            {{
              marketPositions(
                first: 10
                where: {{ market: "{TOKEN_ID}" }}
              ) {{
                id
                netQuantity
                netValue
                valueBought
                valueSold
                user {{
                  id
                }}
              }}
            }}
            """
        }
    
    response = requests.post(ORDERBOOK_URL, headers=headers, json=query, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Check results based on strategy
        if strategy == "enriched":
            trades = data.get('data', {}).get('enrichedOrderFilleds', [])
            print(f"\nFound {len(trades)} trades using enrichedOrderFilleds")
        elif strategy == "orderFilled":
            events = data.get('data', {}).get('orderFilledEvents', [])
            print(f"\nFound {len(events)} order fill events")
        elif strategy == "market_position":
            positions = data.get('data', {}).get('marketPositions', [])
            print(f"\nFound {len(positions)} market positions")
    else:
        print(f"Request failed with status {response.status_code}: {response.text}")

def main():
    print(f"Checking token: {TOKEN_ID}")
    
    # First check token info
    check_token_info()
    
    # Then check orderbook data
    check_orderbook_data()
    
    # Try different trade history queries
    check_trade_history(strategy="enriched")
    check_trade_history(strategy="orderFilled")
    check_trade_history(strategy="market_position")
    
    print("\nDiagnostic check complete")

if __name__ == "__main__":
    main()