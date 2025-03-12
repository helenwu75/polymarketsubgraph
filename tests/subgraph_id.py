#!/usr/bin/env python3
"""
Query Polymarket orderbook subgraph using "clobTokenIds".
Based on the schema, we need to use the ERC1155 token IDs to query market data.
"""

import os
import json
import ast
from dotenv import load_dotenv
from subgrounds import Subgrounds
import pandas as pd
from datetime import datetime

def query_by_token_ids(token_ids_str):
    """
    Query the orderbook subgraph using token IDs.
    
    Args:
        token_ids_str (str): String representation of token IDs list
    
    Returns:
        dict: Combined market data
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")
    
    # Initialize Subgrounds
    sg = Subgrounds()
    
    # Connect to orderbook subgraph
    orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    orderbook_subgraph = sg.load_subgraph(orderbook_url)
    
    # Parse token IDs from string representation
    try:
        # Try to parse the token IDs string - it might be in different formats
        if token_ids_str.startswith('[') and token_ids_str.endswith(']'):
            # It's already in a list-like format
            if '"' in token_ids_str or "'" in token_ids_str:
                # It has quotes, so it might be a JSON string
                try:
                    token_ids = json.loads(token_ids_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try ast.literal_eval
                    token_ids = ast.literal_eval(token_ids_str)
            else:
                # It's a simple string representation of a list
                token_ids = token_ids_str.strip('[]').split(',')
        else:
            # It's a comma-separated string
            token_ids = token_ids_str.split(',')
        
        # Clean up any whitespace
        token_ids = [id.strip().strip('"\'') for id in token_ids]
        
        print(f"Parsed token IDs: {token_ids}")
    except Exception as e:
        print(f"Error parsing token IDs string: {e}")
        print("Using the raw string as a single ID")
        token_ids = [token_ids_str]
    
    # Initialize results dictionary
    results = {
        'token_ids': token_ids,
        'orderbook_data': [],
        'market_data': [],
        'order_events': []
    }
    
    # Query each token ID individually
    for token_id in token_ids:
        print(f"\nQuerying data for token ID: {token_id}")
        
        # 1. Query Orderbook entity
        try:
            orderbook_query = orderbook_subgraph.Query.orderbook(
                id=token_id
            )
            
            orderbook_result = sg.query_df([
                orderbook_query.id,
                orderbook_query.tradesQuantity,
                orderbook_query.buysQuantity,
                orderbook_query.sellsQuantity,
                orderbook_query.collateralVolume,
                orderbook_query.scaledCollateralVolume,
                orderbook_query.scaledCollateralBuyVolume,
                orderbook_query.scaledCollateralSellVolume
            ])
            
            if not orderbook_result.empty and not orderbook_result.iloc[0].isnull().all():
                print("Found orderbook data!")
                orderbook_data = orderbook_result.iloc[0].to_dict()
                results['orderbook_data'].append(orderbook_data)
                
                # Print some key metrics
                print(f"- Trades: {orderbook_data.get('orderbook_tradesQuantity')}")
                print(f"- Volume: {orderbook_data.get('orderbook_scaledCollateralVolume')}")
            else:
                print("No orderbook data found")
        except Exception as e:
            print(f"Error querying orderbook: {e}")
        
        # 2. Query MarketData entity
        try:
            market_data_query = orderbook_subgraph.Query.marketData(
                id=token_id
            )
            
            market_data_result = sg.query_df([
                market_data_query.id,
                market_data_query.condition,
                market_data_query.outcomeIndex,
                market_data_query.priceOrderbook
            ])
            
            if not market_data_result.empty and not market_data_result.iloc[0].isnull().all():
                print("Found market data!")
                market_data = market_data_result.iloc[0].to_dict()
                results['market_data'].append(market_data)
                
                # Print some key information
                print(f"- Condition: {market_data.get('marketData_condition')}")
                print(f"- Outcome Index: {market_data.get('marketData_outcomeIndex')}")
                print(f"- Price: {market_data.get('marketData_priceOrderbook')}")
            else:
                print("No market data found")
        except Exception as e:
            print(f"Error querying market data: {e}")
        
        # 3. Query OrderFilledEvent entities (most recent 5)
        try:
            events_query = orderbook_subgraph.Query.orderFilledEvents(
                first=5,
                orderBy='timestamp',
                orderDirection='desc',
                where={
                    'makerAssetId': token_id
                }
            )
            
            events_result = sg.query_df([
                events_query.id,
                events_query.timestamp,
                events_query.makerAmountFilled,
                events_query.takerAmountFilled,
                events_query.fee
            ])
            
            if not events_result.empty:
                print(f"Found {len(events_result)} order events!")
                # Convert results to list of dictionaries
                events_data = events_result.to_dict('records')
                results['order_events'].extend(events_data)
                
                # Print the most recent event
                recent_event = events_data[0]
                timestamp = int(recent_event.get('orderFilledEvents_timestamp', 0))
                if timestamp > 0:
                    event_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"- Most recent trade: {event_date}")
                    print(f"- Amount: {recent_event.get('orderFilledEvents_makerAmountFilled')} / {recent_event.get('orderFilledEvents_takerAmountFilled')}")
            else:
                print("No order events found")
        except Exception as e:
            print(f"Error querying order events: {e}")
    
    # Calculate summary metrics from all tokens
    if results['orderbook_data']:
        total_volume = sum(float(data.get('orderbook_scaledCollateralVolume', 0) or 0) 
                          for data in results['orderbook_data'])
        total_trades = sum(int(data.get('orderbook_tradesQuantity', 0) or 0) 
                          for data in results['orderbook_data'])
        
        results['summary'] = {
            'total_volume': total_volume,
            'total_trades': total_trades
        }
    
    return results

def main():
    # Trump 2024 presidential election market token IDs
    token_ids_str = '[\"21742633143463906290569050155826241533067272736897614950488156847949938836455\", \"48331043336612883890938759509493159234755048973500640148014422747788308965732\"]'
    
    try:
        results = query_by_token_ids(token_ids_str)
        
        print("\n\nSummary of Results:")
        print("=" * 50)
        
        if results.get('orderbook_data') or results.get('market_data'):
            print("Successfully found data for this market!")
            
            if 'summary' in results:
                print(f"\nTotal Volume: ${results['summary']['total_volume']:,.2f}")
                print(f"Total Trades: {results['summary']['total_trades']:,}")
            
            # Now we can calculate additional metrics like 48-hour volume and daily average
            # (This would require adding timestamp filtering to the queries)
        else:
            print("No data found for this market. This could be because:")
            print("1. The token IDs are not in the correct format")
            print("2. The market is not indexed in this subgraph")
            print("3. The API key doesn't have access to this data")
    except Exception as e:
        print(f"Error executing script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()