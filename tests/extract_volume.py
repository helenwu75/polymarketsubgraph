#!/usr/bin/env python3
"""
Extract key market metrics for a Polymarket election market using token IDs.
This simplified version focuses on reliable metrics and uses estimation for 48-hour volume.
"""

import os
import sys
import json
import ast
import time
from datetime import datetime, timedelta
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

def extract_market_metrics(token_ids, end_date=None, start_date=None, skip_48h=False):
    """
    Extract key metrics for a specific market using token IDs.
    
    Args:
        token_ids (list): List of token IDs
        end_date (str, optional): Market end date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        start_date (str, optional): Market start date in ISO format
        skip_48h (bool): Skip 48-hour volume calculation (faster)
        
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
    
    # Query each token ID for orderbook data
    total_volume = 0
    total_trades = 0
    
    for token_id in token_ids:
        print(f"\nQuerying data for token ID: {token_id}")
        token_data = {'token_id': token_id}
        
        # Query Orderbook entity for total volume and trades
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
                orderbook_query.scaledCollateralVolume
            ])
            
            if not orderbook_result.empty and not orderbook_result.iloc[0].isnull().all():
                print("Found orderbook data!")
                
                # Extract key metrics
                if 'orderbook_tradesQuantity' in orderbook_result.columns:
                    trades_quantity = int(orderbook_result['orderbook_tradesQuantity'].iloc[0] or 0)
                    token_data['trades_quantity'] = trades_quantity
                    total_trades += trades_quantity
                
                if 'orderbook_scaledCollateralVolume' in orderbook_result.columns:
                    volume = float(orderbook_result['orderbook_scaledCollateralVolume'].iloc[0] or 0)
                    token_data['volume'] = volume
                    total_volume += volume
                
                # Extract buys and sells ratio if available
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
        
        # Add token data to metrics
        metrics['tokens_data'].append(token_data)
    
    # Add total volume and trades to metrics
    metrics['total_volume'] = total_volume
    metrics['total_trades'] = total_trades
    
    # Calculate market duration and daily volume if start/end dates are provided
    if start_date and end_date:
        try:
            creation_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            market_end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            market_duration = (market_end_date - creation_date).days
            metrics['market_duration_days'] = market_duration
            
            if market_duration > 0 and total_volume > 0:
                metrics['avg_daily_volume'] = total_volume / market_duration
        except Exception as e:
            print(f"Error calculating market duration: {e}")
    
    # Skip 48-hour volume calculation if requested (it's often problematic)
    if skip_48h or not end_date:
        print("Skipping 48-hour volume calculation")
        
        # If we have market duration, we can estimate 48-hour volume
        if 'market_duration_days' in metrics and metrics['market_duration_days'] > 0:
            # Simple estimate: 48 hours is 2/market_duration_days of the total volume
            # This assumes uniform distribution of volume over time, which is a simplification
            estimated_ratio = 2 / metrics['market_duration_days']
            metrics['volume_final_48h_estimated'] = total_volume * estimated_ratio
            metrics['volume_final_48h_estimated_note'] = "Estimated based on average daily volume"
            
        return metrics
    
    # If we get here, attempt to calculate 48-hour volume
    # (This is the part that often fails for high-volume markets)
    try:
        # Parse end date
        market_end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Calculate timestamp for 48 hours before end (in seconds)
        cutoff_date = market_end_date - timedelta(hours=48)
        cutoff_timestamp = int(cutoff_date.timestamp())
        
        # Query for trades in final 48 hours across all tokens
        volume_48h = 0
        trades_48h = 0
        
        for token_id in token_ids:
            try:
                # Use a smaller batch size with multiple requests to avoid timeouts
                max_events = 100  # Process in smaller batches
                total_token_trades_48h = 0
                total_token_volume_48h = 0
                
                # Start with most recent trades first
                skip = 0
                max_attempts = 2
                
                for attempt in range(max_attempts):
                    try:
                        # Query a batch of trades with timeout protection
                        events_query = orderbook_subgraph.Query.orderFilledEvents(
                            first=max_events,
                            skip=skip,
                            orderBy='timestamp',
                            orderDirection='desc',
                            where={
                                'makerAssetId': token_id
                            }
                        )
                        
                        events_result = sg.query_df([
                            events_query.id,
                            events_query.timestamp,
                            events_query.makerAmountFilled
                        ])
                        
                        if events_result.empty:
                            # No more events to process
                            break
                            
                        # Filter events in Python by timestamp
                        events_48h = events_result[
                            events_result['orderFilledEvents_timestamp'].astype(int) >= cutoff_timestamp
                        ]
                        
                        if not events_48h.empty:
                            # Count trades
                            batch_trades = len(events_48h)
                            total_token_trades_48h += batch_trades
                            
                            # Calculate volume for this batch
                            if 'orderFilledEvents_makerAmountFilled' in events_48h.columns:
                                batch_volume = events_48h['orderFilledEvents_makerAmountFilled'].sum() / 10**6
                                total_token_volume_48h += batch_volume
                        
                        # If we got fewer events than requested, we're done
                        if len(events_result) < max_events:
                            break
                            
                        # Otherwise, prepare for next batch
                        skip += max_events
                        
                    except Exception as e:
                        print(f"Attempt {attempt+1} failed for token {token_id}: {e}")
                        if attempt == max_attempts - 1:
                            # Last attempt failed
                            print(f"Failed to get 48h data for token {token_id} after {max_attempts} attempts")
                        else:
                            # Wait before retrying
                            time.sleep(2)
                
                # Add this token's data to the total
                if total_token_trades_48h > 0:
                    print(f"Found {total_token_trades_48h} trades in final 48 hours for token {token_id}")
                    print(f"Volume in final 48 hours for this token: ${total_token_volume_48h:,.2f}")
                    trades_48h += total_token_trades_48h
                    volume_48h += total_token_volume_48h
                
            except Exception as e:
                print(f"Error querying final 48 hours data for token {token_id}: {e}")
        
        # Add to metrics
        metrics['volume_final_48h'] = volume_48h
        metrics['trades_final_48h'] = trades_48h
        
        # If we have market duration but no valid 48h volume, add the estimate
        if ('volume_final_48h' not in metrics or metrics['volume_final_48h'] == 0) and 'market_duration_days' in metrics and metrics['market_duration_days'] > 0:
            estimated_ratio = 2 / metrics['market_duration_days']
            metrics['volume_final_48h_estimated'] = total_volume * estimated_ratio
            metrics['volume_final_48h_estimated_note'] = "Estimated based on average daily volume"
            
    except Exception as e:
        print(f"Error calculating final 48 hours metrics: {e}")
        
        # If we have market duration, add the estimate
        if 'market_duration_days' in metrics and metrics['market_duration_days'] > 0:
            estimated_ratio = 2 / metrics['market_duration_days']
            metrics['volume_final_48h_estimated'] = total_volume * estimated_ratio
            metrics['volume_final_48h_estimated_note'] = "Estimated based on average daily volume"
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract Polymarket market metrics")
    parser.add_argument("--token-ids", help="Token IDs as JSON string")
    parser.add_argument("--market-id", help="Market ID for reference")
    parser.add_argument("--end-date", help="Market end date (e.g., 2024-11-05T12:00:00Z)")
    parser.add_argument("--start-date", help="Market start date (e.g., 2024-01-04T22:58:00Z)")
    parser.add_argument("--skip-48h", action="store_true", help="Skip 48-hour volume calculation")
    args = parser.parse_args()
    
    # Default values (Trump 2024 presidential election market)
    token_ids_str = args.token_ids or '[\"21742633143463906290569050155826241533067272736897614950488156847949938836455\", \"48331043336612883890938759509493159234755048973500640148014422747788308965732\"]'
    end_date = args.end_date or '2024-11-05T12:00:00Z'  # Election day
    start_date = args.start_date or '2024-01-04T22:58:00Z'  # From original JSON
    skip_48h = args.skip_48h
    
    try:
        # Parse token IDs
        token_ids = parse_token_ids(token_ids_str)
        if not token_ids:
            print("Error: No valid token IDs found")
            return
        
        print(f"Parsed token IDs: {token_ids}")
        
        # Extract metrics
        metrics = extract_market_metrics(token_ids, end_date, start_date, skip_48h)
        
        # Print results
        print("\nMarket Metrics Summary:")
        print("=" * 50)
        
        if args.market_id:
            print(f"Market ID: {args.market_id}")
        
        if 'total_volume' in metrics:
            print(f"Total Volume: ${metrics['total_volume']:,.2f}")
        
        if 'total_trades' in metrics:
            print(f"Total Trades: {metrics['total_trades']:,}")
        
        if 'market_duration_days' in metrics:
            print(f"Market Duration: {metrics['market_duration_days']} days")
        
        if 'avg_daily_volume' in metrics:
            print(f"Avg Daily Volume: ${metrics['avg_daily_volume']:,.2f}")
        
        # Print 48-hour volume (actual or estimated)
        if 'volume_final_48h' in metrics and metrics['volume_final_48h'] > 0:
            print(f"Volume in Final 48 Hours: ${metrics['volume_final_48h']:,.2f}")
            
            if 'trades_final_48h' in metrics:
                print(f"Trades in Final 48 Hours: {metrics['trades_final_48h']:,}")
            
            if metrics['total_volume'] > 0:
                percentage = (metrics['volume_final_48h'] / metrics['total_volume']) * 100
                print(f"Final 48h Volume as % of Total: {percentage:.2f}%")
        elif 'volume_final_48h_estimated' in metrics:
            print(f"Est. Volume in Final 48 Hours: ${metrics['volume_final_48h_estimated']:,.2f}")
            print(f"Note: {metrics['volume_final_48h_estimated_note']}")
            
            if metrics['total_volume'] > 0:
                percentage = (metrics['volume_final_48h_estimated'] / metrics['total_volume']) * 100
                print(f"Est. Final 48h Volume as % of Total: {percentage:.2f}%")
        else:
            print("Volume in Final 48 Hours: Not available")
        
        # Print data for individual tokens
        if 'tokens_data' in metrics:
            print("\nIndividual Token Data:")
            for idx, token_data in enumerate(metrics['tokens_data']):
                print(f"\nToken {idx+1}: {token_data.get('token_id')}")
                if 'volume' in token_data:
                    print(f"- Volume: ${token_data['volume']:,.2f}")
                if 'trades_quantity' in token_data:
                    print(f"- Trades: {token_data['trades_quantity']:,}")
                if 'buy_sell_ratio' in token_data:
                    print(f"- Buy/Sell Ratio: {token_data['buy_sell_ratio']:.2f}")
        
    except Exception as e:
        print(f"Error executing script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()