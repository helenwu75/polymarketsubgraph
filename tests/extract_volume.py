#!/usr/bin/env python3
"""
Extract accurate market metrics for a Polymarket market using token IDs.
This version sums individual order amounts directly from order filled events,
matching the approach used in Polymarket.
"""

import os
import sys
import json
import ast
import time
from datetime import datetime
from dotenv import load_dotenv
from subgrounds import Subgrounds
import pandas as pd
import argparse

def parse_token_ids(token_ids_str):
    """
    Parse token IDs from a string representation.
    
    Args:
        token_ids_str (str): String representation of token IDs list
        
    Returns:
        list: List of token IDs
    """
    if not token_ids_str or token_ids_str == 'nan' or token_ids_str == 'None':
        return []
        
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
        return token_ids
    except Exception as e:
        print(f"Error parsing token IDs string: {e}")
        return [token_ids_str]  # Return as a single ID

def extract_market_metrics(token_ids, end_date=None, start_date=None):
    """
    Extract key metrics for a specific market using token IDs.
    This version calculates volume based on individual order filled events,
    matching the algorithm used by Polymarket.
    
    Args:
        token_ids (list): List of token IDs
        end_date (str, optional): Market end date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        start_date (str, optional): Market start date in ISO format
        
    Returns:
        dict: Market metrics
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
    
    # Initialize metrics dictionary
    metrics = {
        'token_ids': token_ids,
        'tokens_data': []
    }
    
    # Query each token ID for data
    total_trades = 0
    total_buy_usdc = 0
    total_sell_usdc = 0
    
    for token_id in token_ids:
        print(f"\nQuerying data for token ID: {token_id}")
        token_data = {'token_id': token_id}
        
        # Get basic orderbook stats
        try:
            orderbook_query = orderbook_subgraph.Query.orderbook(
                id=token_id
            )
            
            orderbook_result = sg.query_df([
                orderbook_query.id,
                orderbook_query.tradesQuantity,
                orderbook_query.buysQuantity,
                orderbook_query.sellsQuantity
            ])
            
            if not orderbook_result.empty and not orderbook_result.iloc[0].isnull().all():
                print("Found orderbook data!")
                
                # Extract trade counts
                if 'orderbook_tradesQuantity' in orderbook_result.columns:
                    trades_quantity = int(orderbook_result['orderbook_tradesQuantity'].iloc[0] or 0)
                    token_data['trades_quantity'] = trades_quantity
                    total_trades += trades_quantity
                
                # Extract buys and sells quantity
                if 'orderbook_buysQuantity' in orderbook_result.columns and 'orderbook_sellsQuantity' in orderbook_result.columns:
                    buys = int(orderbook_result['orderbook_buysQuantity'].iloc[0] or 0)
                    sells = int(orderbook_result['orderbook_sellsQuantity'].iloc[0] or 0)
                    token_data['buys_quantity'] = buys
                    token_data['sells_quantity'] = sells
                    
                    if sells > 0:
                        token_data['buy_sell_ratio'] = buys / sells
                    else:
                        token_data['buy_sell_ratio'] = buys if buys > 0 else 0
            else:
                print("No orderbook data found")
        except Exception as e:
            print(f"Error querying orderbook: {e}")
        
                    # Get buy/sell USDC amounts from Global entity for better performance
        try:
            print("Querying pre-aggregated USDC volumes from orderbook entity...")
            
            # Query the orderbook entity for pre-aggregated volume data
            # This is MUCH faster than processing individual trades
            orderbook_query = orderbook_subgraph.Query.orderbook(
                id=token_id
            )
            
            volume_result = sg.query_df([
                orderbook_query.id,
                # Raw volume (needs to be divided by 10^6)
                orderbook_query.collateralBuyVolume,
                orderbook_query.collateralSellVolume,
                # Pre-scaled volume
                orderbook_query.scaledCollateralBuyVolume,
                orderbook_query.scaledCollateralSellVolume
            ])
            
            if not volume_result.empty:
                # Use scaled volumes if available (already divided by 10^6)
                if 'orderbook_scaledCollateralBuyVolume' in volume_result.columns:
                    token_buy_usdc = float(volume_result['orderbook_scaledCollateralBuyVolume'].iloc[0] or 0)
                elif 'orderbook_collateralBuyVolume' in volume_result.columns:
                    token_buy_usdc = float(volume_result['orderbook_collateralBuyVolume'].iloc[0] or 0) / 10**6
                else:
                    token_buy_usdc = 0
                    
                if 'orderbook_scaledCollateralSellVolume' in volume_result.columns:
                    token_sell_usdc = float(volume_result['orderbook_scaledCollateralSellVolume'].iloc[0] or 0)
                elif 'orderbook_collateralSellVolume' in volume_result.columns:
                    token_sell_usdc = float(volume_result['orderbook_collateralSellVolume'].iloc[0] or 0) / 10**6
                else:
                    token_sell_usdc = 0
            
            # Add volume data to token metrics
            token_data['buy_usdc'] = token_buy_usdc
            token_data['sell_usdc'] = token_sell_usdc
            token_data['total_usdc'] = token_buy_usdc + token_sell_usdc
            
            # Add to global totals
            total_buy_usdc += token_buy_usdc
            total_sell_usdc += token_sell_usdc
            
            print(f"Total trades processed: {processed_trades}")
            print(f"Buy USDC: ${token_buy_usdc:,.2f}")
            print(f"Sell USDC: ${token_sell_usdc:,.2f}")
            print(f"Total USDC: ${token_data['total_usdc']:,.2f}")
            
        except Exception as e:
            print(f"Error calculating USDC volume: {e}")
            import traceback
            traceback.print_exc()
        
        # Add token data to metrics
        metrics['tokens_data'].append(token_data)
    
    # Add total metrics
    metrics['total_buy_usdc'] = total_buy_usdc
    metrics['total_sell_usdc'] = total_sell_usdc
    metrics['total_usdc'] = total_buy_usdc + total_sell_usdc
    metrics['total_trades'] = total_trades
    
    # Calculate market duration and daily volume if start/end dates are provided
    if start_date and end_date:
        try:
            creation_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            market_end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            market_duration = (market_end_date - creation_date).days
            metrics['market_duration_days'] = market_duration
            
            if market_duration > 0 and metrics['total_usdc'] > 0:
                metrics['avg_daily_volume'] = metrics['total_usdc'] / market_duration
        except Exception as e:
            print(f"Error calculating market duration: {e}")
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract Polymarket market metrics")
    parser.add_argument("--token-ids", help="Token IDs as JSON string")
    parser.add_argument("--market-id", help="Market ID for reference")
    parser.add_argument("--end-date", help="Market end date (e.g., 2024-11-05T12:00:00Z)")
    parser.add_argument("--start-date", help="Market start date (e.g., 2024-01-04T22:58:00Z)")
    args = parser.parse_args()
    
    # Default values (Trump 2024 presidential election market)
    token_ids_str = args.token_ids or '[\"21742633143463906290569050155826241533067272736897614950488156847949938836455\", \"48331043336612883890938759509493159234755048973500640148014422747788308965732\"]'
    end_date = args.end_date or '2024-11-05T12:00:00Z'  # Election day
    start_date = args.start_date or '2024-01-04T22:58:00Z'  # From original JSON
    
    try:
        # Parse token IDs
        token_ids = parse_token_ids(token_ids_str)
        if not token_ids:
            print("Error: No valid token IDs found")
            return
        
        print(f"Parsed token IDs: {token_ids}")
        
        # Extract metrics
        metrics = extract_market_metrics(token_ids, end_date, start_date)
        
        # Print results
        print("\nMarket Metrics Summary:")
        print("=" * 50)
        
        if args.market_id:
            print(f"Market ID: {args.market_id}")
        
        # Print volume metrics
        print(f"Total Buy USDC: ${metrics['total_buy_usdc']:,.2f}")
        print(f"Total Sell USDC: ${metrics['total_sell_usdc']:,.2f}")
        print(f"Total USDC Volume: ${metrics['total_usdc']:,.2f}")
        
        if 'total_trades' in metrics:
            print(f"Total Trades: {metrics['total_trades']:,}")
        
        if 'market_duration_days' in metrics:
            print(f"Market Duration: {metrics['market_duration_days']} days")
        
        if 'avg_daily_volume' in metrics:
            print(f"Avg Daily Volume: ${metrics['avg_daily_volume']:,.2f}")
        
        # Print data for individual tokens
        if 'tokens_data' in metrics:
            print("\nIndividual Token Data:")
            for idx, token_data in enumerate(metrics['tokens_data']):
                print(f"\nToken {idx+1}: {token_data.get('token_id')}")
                
                if 'buy_usdc' in token_data and 'sell_usdc' in token_data:
                    print(f"- Buy USDC: ${token_data['buy_usdc']:,.2f}")
                    print(f"- Sell USDC: ${token_data['sell_usdc']:,.2f}")
                    print(f"- Total USDC: ${token_data.get('total_usdc', 0):,.2f}")
                    
                if 'trades_quantity' in token_data:
                    print(f"- Trades: {token_data['trades_quantity']:,}")
                    
                if 'buy_sell_ratio' in token_data:
                    print(f"- Buy/Sell Ratio: {token_data['buy_sell_ratio']:.2f}")
        
        # Save results to JSON file
        output_file = f"market_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nDetailed metrics saved to: {output_file}")
        
    except Exception as e:
        print(f"Error executing script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()