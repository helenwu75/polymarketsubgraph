#!/usr/bin/env python3
"""
Simple script to calculate total volume in terms of shares (contracts) traded
on Polymarket markets, rather than USDC value.
"""

import os
import json
import ast
from dotenv import load_dotenv
from subgrounds import Subgrounds
import argparse

def parse_token_ids(token_ids_str):
    """Parse token IDs from a string representation"""
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
        print(f"Error parsing token IDs: {e}")
        return [token_ids_str]

def calculate_share_volume(token_ids):
    """
    Calculate the total volume in terms of shares (contracts) traded
    
    Args:
        token_ids (list): List of token IDs for the market
        
    Returns:
        dict: Market metrics with share volumes
    """
    # Load API key
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")
    
    # Initialize Subgrounds
    sg = Subgrounds()
    
    # Connect to orderbook subgraph
    orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    orderbook_subgraph = sg.load_subgraph(orderbook_url)
    
    # Initialize metrics
    metrics = {
        'token_ids': token_ids,
        'tokens_data': [],
        'total_trades': 0,
        'total_shares_volume': 0
    }
    
    for token_id in token_ids:
        print(f"\nQuerying data for token ID: {token_id}")
        token_data = {'token_id': token_id}
        
        try:
            # Query orderbook for raw volume data
            orderbook_query = orderbook_subgraph.Query.orderbook(
                id=token_id
            )
            
            orderbook_result = sg.query_df([
                orderbook_query.id,
                orderbook_query.tradesQuantity,
                orderbook_query.collateralBuyVolume,
                orderbook_query.collateralSellVolume,
                orderbook_query.collateralVolume
            ])
            
            if not orderbook_result.empty and not orderbook_result.iloc[0].isnull().all():
                print(f"Found orderbook data for token {token_id}")
                
                # Get trade count
                if 'orderbook_tradesQuantity' in orderbook_result.columns:
                    trades = int(orderbook_result['orderbook_tradesQuantity'].iloc[0] or 0)
                    token_data['trades'] = trades
                    metrics['total_trades'] += trades
                    print(f"Total trades: {trades:,}")
                
                # Get raw volume data
                raw_buy_volume = 0
                raw_sell_volume = 0
                
                if 'orderbook_collateralBuyVolume' in orderbook_result.columns:
                    raw_buy_volume = float(orderbook_result['orderbook_collateralBuyVolume'].iloc[0] or 0)
                    token_data['raw_buy_volume'] = raw_buy_volume
                
                if 'orderbook_collateralSellVolume' in orderbook_result.columns:
                    raw_sell_volume = float(orderbook_result['orderbook_collateralSellVolume'].iloc[0] or 0)
                    token_data['raw_sell_volume'] = raw_sell_volume
                
                # Calculate shares volume using different methods
                
                # Method 1: Estimate share count by dividing by 10^8
                # This assumes average price of around 0.01 USDC per share
                shares_volume_1 = (raw_buy_volume + raw_sell_volume) / 10**8
                token_data['shares_volume_1'] = shares_volume_1
                print(f"Estimated shares (method 1): {shares_volume_1:,.2f}")
                
                # Method 2: Use trade count as proxy for share count
                # This assumes each trade is approximately 1 share
                shares_volume_2 = trades
                token_data['shares_volume_2'] = shares_volume_2
                print(f"Estimated shares (method 2): {shares_volume_2:,}")
                
                # Method 3: Another scale factor (10^7) for comparison
                shares_volume_3 = (raw_buy_volume + raw_sell_volume) / 10**7
                token_data['shares_volume_3'] = shares_volume_3
                print(f"Estimated shares (method 3): {shares_volume_3:,.2f}")
                
                # Use method 1 as our primary method
                token_data['shares_volume'] = shares_volume_1
                metrics['total_shares_volume'] += shares_volume_1
            else:
                print(f"No data found for token {token_id}")
        
        except Exception as e:
            print(f"Error querying data for token {token_id}: {e}")
        
        # Add token data to metrics
        metrics['tokens_data'].append(token_data)
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate share volume for Polymarket markets")
    parser.add_argument("--token-ids", help="Token IDs as JSON string")
    parser.add_argument("--market-id", help="Market ID for reference")
    args = parser.parse_args()
    
    # Default values (Trump 2024 election market)
    token_ids_str = args.token_ids or '[\"21742633143463906290569050155826241533067272736897614950488156847949938836455\", \"48331043336612883890938759509493159234755048973500640148014422747788308965732\"]'
    
    try:
        # Parse token IDs
        token_ids = parse_token_ids(token_ids_str)
        if not token_ids:
            print("Error: No valid token IDs found")
            return
        
        print(f"Parsed token IDs: {token_ids}")
        
        # Calculate share volume
        metrics = calculate_share_volume(token_ids)
        
        # Print results
        print("\n==========================================")
        print("SHARE VOLUME RESULTS")
        print("==========================================")
        
        if args.market_id:
            print(f"Market ID: {args.market_id}")
        
        print(f"Total Trades: {metrics['total_trades']:,}")
        print(f"Total Share Volume: {metrics['total_shares_volume']:,.2f}")
        
        # Print individual token data
        print("\nIndividual Token Data:")
        for idx, token_data in enumerate(metrics['tokens_data']):
            print(f"\nToken {idx+1}: {token_data.get('token_id')}")
            print(f"- Trades: {token_data.get('trades', 0):,}")
            print(f"- Share Volume: {token_data.get('shares_volume', 0):,.2f}")
        
        print("\nNote: Share volume is estimated based on raw collateral volume divided by 10^8")
        print("This is an approximation and may not reflect exact share counts")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()