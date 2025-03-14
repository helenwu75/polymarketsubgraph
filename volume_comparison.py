#!/usr/bin/env python3
"""
Direct Subgraph Polymarket Volume Comparison

This script fetches volume data directly from the Polymarket subgraph for four different data sources:
1. enrichedOrderFilled events
2. orderFilledEvent records
3. orderbook entity data
4. ordersMatchedEvent records

Usage:
    python volume_comparison.py --token-id 79316691944049488812500733050438507204613781002222375264046442941003895009475
"""

import os
import argparse
import pandas as pd
import requests
from dotenv import load_dotenv

def fetch_enriched_order_data(token_id, api_key):
    """Fetch enrichedOrderFilled data directly from the subgraph."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    query = {
        'query': f"""
        {{
          enrichedOrderFilleds(
            first: 1000
            where: {{ market: "{token_id}" }}
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
    
    print(f"Fetching enrichedOrderFilled data for token {token_id}...")
    
    try:
        response = requests.post(url, headers=headers, json=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('data', {}).get('enrichedOrderFilleds', [])
            
            if events:
                print(f"Fetched {len(events)} enrichedOrderFilled events")
                return pd.DataFrame(events)
            else:
                print("No enrichedOrderFilled events found")
                return None
        else:
            print(f"API request failed with status code {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Error fetching enrichedOrderFilled data: {e}")
        return None

def fetch_order_filled_events(token_id, api_key):
    """Fetch orderFilledEvent data directly from the subgraph."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # First try with token as maker asset
    query = {
        'query': f"""
        {{
          orderFilledEvents(
            first: 1000
            where: {{ makerAssetId: "{token_id}" }}
          ) {{
            id
            timestamp
            makerAssetId
            takerAssetId
            makerAmountFilled
            takerAmountFilled
            fee
          }}
        }}
        """
    }
    
    print(f"Fetching orderFilledEvent data for token {token_id} as maker asset...")
    
    all_events = []
    
    try:
        response = requests.post(url, headers=headers, json=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('data', {}).get('orderFilledEvents', [])
            
            if events:
                print(f"Fetched {len(events)} orderFilledEvents (token as maker)")
                all_events.extend(events)
        else:
            print(f"API request failed with status code {response.status_code}")
    
    except Exception as e:
        print(f"Error fetching orderFilledEvent data (maker): {e}")
    
    # Then try with token as taker asset
    query = {
        'query': f"""
        {{
          orderFilledEvents(
            first: 1000
            where: {{ takerAssetId: "{token_id}" }}
          ) {{
            id
            timestamp
            makerAssetId
            takerAssetId
            makerAmountFilled
            takerAmountFilled
            fee
          }}
        }}
        """
    }
    
    print(f"Fetching orderFilledEvent data for token {token_id} as taker asset...")
    
    try:
        response = requests.post(url, headers=headers, json=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('data', {}).get('orderFilledEvents', [])
            
            if events:
                print(f"Fetched {len(events)} orderFilledEvents (token as taker)")
                
                # Add only non-duplicate events
                event_ids = {e['id'] for e in all_events}
                new_events = [e for e in events if e['id'] not in event_ids]
                
                print(f"Adding {len(new_events)} unique events")
                all_events.extend(new_events)
        else:
            print(f"API request failed with status code {response.status_code}")
    
    except Exception as e:
        print(f"Error fetching orderFilledEvent data (taker): {e}")
    
    if all_events:
        df = pd.DataFrame(all_events)
        df['token_id'] = token_id  # Add token ID for reference
        return df
    else:
        print("No orderFilledEvents found")
        return None

def fetch_orderbook_data(token_id, api_key):
    """Fetch orderbook entity data directly from the subgraph."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    query = {
        'query': f"""
        {{
          orderbook(id: "{token_id}") {{
            id
            collateralVolume
            scaledCollateralVolume
            collateralBuyVolume
            scaledCollateralBuyVolume
            collateralSellVolume
            scaledCollateralSellVolume
            tradesQuantity
            buysQuantity
            sellsQuantity
          }}
        }}
        """
    }
    
    print(f"Fetching orderbook entity data for token {token_id}...")
    
    try:
        response = requests.post(url, headers=headers, json=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            orderbook = data.get('data', {}).get('orderbook')
            
            if orderbook:
                print(f"Fetched orderbook data: {orderbook['id']}")
                return orderbook
            else:
                print("No orderbook entity found")
                return None
        else:
            print(f"API request failed with status code {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Error fetching orderbook data: {e}")
        return None

def fetch_orders_matched_events(token_id, api_key):
    """Fetch ordersMatchedEvent data directly from the subgraph."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Try with makerAssetID (uppercase ID)
    query = {
        'query': f"""
        {{
          ordersMatchedEvents(
            first: 1000
            where: {{ makerAssetID: "{token_id}" }}
          ) {{
            id
            timestamp
            makerAssetID
            takerAssetID
            makerAmountFilled
            takerAmountFilled
          }}
        }}
        """
    }
    
    print(f"Fetching ordersMatchedEvents with token {token_id} as maker asset...")
    
    all_events = []
    
    try:
        response = requests.post(url, headers=headers, json=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for errors that might indicate wrong field names
            if 'errors' in data:
                print("Error with initial query, trying alternative field names")
                # Try with lowercase field names
                query = {
                    'query': f"""
                    {{
                      ordersMatchedEvents(
                        first: 1000
                        where: {{ makerAssetId: "{token_id}" }}
                      ) {{
                        id
                        timestamp
                        makerAssetId
                        takerAssetId
                        makerAmountFilled
                        takerAmountFilled
                      }}
                    }}
                    """
                }
                response = requests.post(url, headers=headers, json=query, timeout=30)
                data = response.json()
            
            events = data.get('data', {}).get('ordersMatchedEvents', [])
            
            if events:
                print(f"Fetched {len(events)} ordersMatchedEvents (token as maker)")
                all_events.extend(events)
            
            # Now try with token as taker asset
            # First determine which field name format was successful
            if 'errors' not in data:
                taker_query = {
                    'query': f"""
                    {{
                      ordersMatchedEvents(
                        first: 1000
                        where: {{ takerAssetID: "{token_id}" }}
                      ) {{
                        id
                        timestamp
                        makerAssetID
                        takerAssetID
                        makerAmountFilled
                        takerAmountFilled
                      }}
                    }}
                    """
                }
            else:
                taker_query = {
                    'query': f"""
                    {{
                      ordersMatchedEvents(
                        first: 1000
                        where: {{ takerAssetId: "{token_id}" }}
                      ) {{
                        id
                        timestamp
                        makerAssetId
                        takerAssetId
                        makerAmountFilled
                        takerAmountFilled
                      }}
                    }}
                    """
                }
            
            print(f"Fetching ordersMatchedEvents with token {token_id} as taker asset...")
            
            response = requests.post(url, headers=headers, json=taker_query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('data', {}).get('ordersMatchedEvents', [])
                
                if events:
                    print(f"Fetched {len(events)} ordersMatchedEvents (token as taker)")
                    
                    # Add only non-duplicate events
                    event_ids = {e['id'] for e in all_events}
                    new_events = [e for e in events if e['id'] not in event_ids]
                    
                    print(f"Adding {len(new_events)} unique events")
                    all_events.extend(new_events)
        else:
            print(f"API request failed with status code {response.status_code}")
    
    except Exception as e:
        print(f"Error fetching ordersMatchedEvent data: {e}")
    
    if all_events:
        df = pd.DataFrame(all_events)
        df['token_id'] = token_id  # Add token ID for reference
        return df
    else:
        print("No ordersMatchedEvents found")
        return None

def calculate_enriched_volumes(df):
    """Calculate volumes from enrichedOrderFilled data."""
    if df is None or df.empty:
        return None
    
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    total_volume = df['size'].sum()
    
    buy_volume = df[df['side'] == 'Buy']['size'].sum()
    sell_volume = df[df['side'] == 'Sell']['size'].sum()
    
    buy_count = len(df[df['side'] == 'Buy'])
    sell_count = len(df[df['side'] == 'Sell'])
    
    print("\nEnrichedOrderFilled Volume Summary:")
    print(f"  Total Volume: {total_volume:,.2f}")
    print(f"  Buy Volume: {buy_volume:,.2f} ({buy_count} trades)")
    print(f"  Sell Volume: {sell_volume:,.2f} ({sell_count} trades)")
    
    return {
        'total_volume': total_volume,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'buy_count': buy_count,
        'sell_count': sell_count
    }

def calculate_order_filled_volumes(df):
    """Calculate volumes from orderFilledEvent data."""
    if df is None or df.empty:
        return None
    
    try:
        df['makerAmountFilled'] = pd.to_numeric(df['makerAmountFilled'], errors='coerce')
        df['takerAmountFilled'] = pd.to_numeric(df['takerAmountFilled'], errors='coerce')
        
        token_id = df['token_id'].iloc[0] if 'token_id' in df.columns else None
        
        maker_volume = df['makerAmountFilled'].sum()
        taker_volume = df['takerAmountFilled'].sum()
        
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        
        # If we can identify the token properly
        if token_id and 'makerAssetId' in df.columns and 'takerAssetId' in df.columns:
            for _, row in df.iterrows():
                is_maker_token = str(row['makerAssetId']) == str(token_id)
                is_taker_token = str(row['takerAssetId']) == str(token_id)
                
                if is_maker_token:
                    sell_volume += float(row['makerAmountFilled'])
                    sell_count += 1
                elif is_taker_token:
                    buy_volume += float(row['takerAmountFilled'])
                    buy_count += 1
            
            token_volume = buy_volume + sell_volume
        else:
            token_volume = maker_volume + taker_volume
        
        print("\nOrderFilledEvent Volume Summary:")
        print(f"  Total Volume: {token_volume:,.2f}")
        print(f"  Maker Volume: {maker_volume:,.2f}")
        print(f"  Taker Volume: {taker_volume:,.2f}")
        
        if buy_volume > 0 or sell_volume > 0:
            print(f"  Buy Volume: {buy_volume:,.2f} ({buy_count} trades)")
            print(f"  Sell Volume: {sell_volume:,.2f} ({sell_count} trades)")
        
        return {
            'total_volume': token_volume,
            'maker_volume': maker_volume,
            'taker_volume': taker_volume,
            'buy_volume': buy_volume if buy_volume > 0 else None,
            'sell_volume': sell_volume if sell_volume > 0 else None
        }
    
    except Exception as e:
        print(f"Error calculating orderFilledEvent volumes: {e}")
        return None

def extract_orderbook_volumes(data):
    """Extract volume data from orderbook entity."""
    if data is None:
        return None
    
    volumes = {}
    
    if 'scaledCollateralVolume' in data:
        volumes['total_volume'] = float(data['scaledCollateralVolume'])
    elif 'collateralVolume' in data:
        volumes['total_volume'] = float(data['collateralVolume'])
    
    if 'scaledCollateralBuyVolume' in data:
        volumes['buy_volume'] = float(data['scaledCollateralBuyVolume'])
    elif 'collateralBuyVolume' in data:
        volumes['buy_volume'] = float(data['collateralBuyVolume'])
    
    if 'scaledCollateralSellVolume' in data:
        volumes['sell_volume'] = float(data['scaledCollateralSellVolume'])
    elif 'collateralSellVolume' in data:
        volumes['sell_volume'] = float(data['collateralSellVolume'])
    
    if 'tradesQuantity' in data:
        volumes['trade_count'] = int(data['tradesQuantity'])
    
    print("\nOrderbook Entity Volume Summary:")
    print(f"  Total Volume: {volumes.get('total_volume', 'N/A'):,.2f}")
    print(f"  Buy Volume: {volumes.get('buy_volume', 'N/A'):,.2f}")
    print(f"  Sell Volume: {volumes.get('sell_volume', 'N/A'):,.2f}")
    print(f"  Trade Count: {volumes.get('trade_count', 'N/A'):,}")
    
    return volumes

def calculate_orders_matched_volumes(df):
    """Calculate volumes from ordersMatchedEvent data."""
    if df is None or df.empty:
        return None
    
    try:
        # First determine which field names are used (capital ID vs lowercase id)
        maker_col = 'makerAssetID' if 'makerAssetID' in df.columns else 'makerAssetId'
        taker_col = 'takerAssetID' if 'takerAssetID' in df.columns else 'takerAssetId'
        
        if maker_col not in df.columns or taker_col not in df.columns:
            print(f"Required asset columns not found. Available columns: {df.columns.tolist()}")
            return None
        
        df['makerAmountFilled'] = pd.to_numeric(df['makerAmountFilled'], errors='coerce')
        df['takerAmountFilled'] = pd.to_numeric(df['takerAmountFilled'], errors='coerce')
        
        maker_amount = df['makerAmountFilled'].sum()
        taker_amount = df['takerAmountFilled'].sum()
        
        token_id = df['token_id'].iloc[0] if 'token_id' in df.columns else None
        
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        
        # If we can identify the token properly
        if token_id:
            for _, row in df.iterrows():
                is_maker_token = str(row[maker_col]) == str(token_id)
                is_taker_token = str(row[taker_col]) == str(token_id)
                
                if is_maker_token:
                    sell_volume += float(row['makerAmountFilled'])
                    sell_count += 1
                elif is_taker_token:
                    buy_volume += float(row['takerAmountFilled'])
                    buy_count += 1
            
            token_volume = buy_volume + sell_volume
        else:
            token_volume = maker_amount + taker_amount
        
        print("\nOrdersMatchedEvent Volume Summary:")
        print(f"  Total Volume: {token_volume:,.2f}")
        print(f"  Maker Amount: {maker_amount:,.2f}")
        print(f"  Taker Amount: {taker_amount:,.2f}")
        
        if buy_volume > 0 or sell_volume > 0:
            print(f"  Buy Volume: {buy_volume:,.2f} ({buy_count} events)")
            print(f"  Sell Volume: {sell_volume:,.2f} ({sell_count} events)")
        
        return {
            'total_volume': token_volume,
            'maker_amount': maker_amount,
            'taker_amount': taker_amount,
            'buy_volume': buy_volume if buy_volume > 0 else None,
            'sell_volume': sell_volume if sell_volume > 0 else None
        }
    
    except Exception as e:
        print(f"Error calculating ordersMatchedEvent volumes: {e}")
        return None

def compare_volumes(enriched, order_filled, orderbook, orders_matched):
    """Compare volumes from all four data sources."""
    print("\n=== VOLUME COMPARISON SUMMARY ===")
    
    # Create comparison table
    data = {
        'Data Source': ['EnrichedOrderFilled', 'OrderFilledEvent', 'Orderbook Entity', 'OrdersMatchedEvent'],
        'Total Volume': [
            enriched['total_volume'] if enriched else None,
            order_filled['total_volume'] if order_filled else None,
            orderbook.get('total_volume') if orderbook else None,
            orders_matched['total_volume'] if orders_matched else None
        ]
    }
    
    # Add buy/sell volumes if available
    buy_volumes = [
        enriched['buy_volume'] if enriched else None,
        order_filled.get('buy_volume') if order_filled else None,
        orderbook.get('buy_volume') if orderbook else None,
        orders_matched.get('buy_volume') if orders_matched else None
    ]
    
    sell_volumes = [
        enriched['sell_volume'] if enriched else None,
        order_filled.get('sell_volume') if order_filled else None,
        orderbook.get('sell_volume') if orderbook else None,
        orders_matched.get('sell_volume') if orders_matched else None
    ]
    
    # Only add buy/sell columns if we have some data
    if any(vol is not None for vol in buy_volumes):
        data['Buy Volume'] = buy_volumes
    
    if any(vol is not None for vol in sell_volumes):
        data['Sell Volume'] = sell_volumes
    
    # Create and display comparison dataframe
    comparison_df = pd.DataFrame(data)
    comparison_df.set_index('Data Source', inplace=True)
    print(comparison_df.to_string())
    
    # Calculate differences using EnrichedOrderFilled as baseline
    if enriched and enriched['total_volume'] > 0:
        baseline = enriched['total_volume']
        print("\nDifferences from EnrichedOrderFilled:")
        
        # OrderFilledEvent difference
        if order_filled and order_filled['total_volume'] is not None:
            diff = (order_filled['total_volume'] / baseline - 1) * 100
            print(f"  OrderFilledEvent: {diff:+.2f}%")
        
        # Orderbook difference
        if orderbook and orderbook.get('total_volume') is not None:
            diff = (orderbook['total_volume'] / baseline - 1) * 100
            print(f"  Orderbook Entity: {diff:+.2f}%")
        
        # OrdersMatchedEvent difference
        if orders_matched and orders_matched['total_volume'] is not None:
            diff = (orders_matched['total_volume'] / baseline - 1) * 100
            print(f"  OrdersMatchedEvent: {diff:+.2f}%")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description="Compare Polymarket volume data across different sources")
    parser.add_argument("--token-id", required=True, help="Token ID to analyze")
    args = parser.parse_args()
    
    # Load API key from environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        print("ERROR: GRAPH_API_KEY not found in environment variables")
        print("Please create a .env file with your Graph API key:")
        print("GRAPH_API_KEY=your_api_key_here")
        return
    
    print(f"\n=== Polymarket Volume Analysis for Token: {args.token_id} ===\n")
    
    # Fetch data directly from the subgraph
    enriched_df = fetch_enriched_order_data(args.token_id, api_key)
    order_filled_df = fetch_order_filled_events(args.token_id, api_key)
    orderbook_data = fetch_orderbook_data(args.token_id, api_key)
    orders_matched_df = fetch_orders_matched_events(args.token_id, api_key)
    
    # Calculate volumes from each source
    enriched_volumes = calculate_enriched_volumes(enriched_df)
    order_filled_volumes = calculate_order_filled_volumes(order_filled_df)
    orderbook_volumes = extract_orderbook_volumes(orderbook_data)
    orders_matched_volumes = calculate_orders_matched_volumes(orders_matched_df)
    
    # Compare volumes
    compare_volumes(
        enriched_volumes,
        order_filled_volumes,
        orderbook_volumes,
        orders_matched_volumes
    )

if __name__ == "__main__":
    main()