#!/usr/bin/env python3
"""
Enhanced Market Metrics Extractor for Polymarket Election Markets

This script extracts comprehensive metrics for Polymarket markets, including:
- Average daily volume
- Days between market creation and election event
- Number of unique traders
- Trader-to-volume ratio (average volume per trader)
- Trading frequency (average trades per trader)
- Buy/sell ratio

Usage:
    python extract_metrics.py --market-id <market_id> --token-ids <token_ids_str> 
                             --start-date <start_date> --end-date <end_date>
"""

import os
import sys
import json
import ast
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
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

def get_market_creation_date(market_id: str) -> Optional[datetime]:
    """
    Get the market creation date from the activity subgraph.
    
    Args:
        market_id (str): Market ID (FixedProductMarketMaker address)
        
    Returns:
        Optional[datetime]: Market creation date or None if not found
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")
    
    # Initialize Subgrounds
    sg = Subgrounds()
    
    # Connect to activity subgraph (which has creation timestamps)
    activity_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
    activity_subgraph = sg.load_subgraph(activity_url)
    
    try:
        # Query market creation timestamp
        query = activity_subgraph.Query.fixedProductMarketMaker(
            id=market_id.lower()  # Ensure lowercase for Ethereum addresses
        )
        
        result = sg.query_df([
            query.id,
            query.creationTimestamp
        ])
        
        if not result.empty and 'fixedProductMarketMaker_creationTimestamp' in result.columns:
            timestamp = int(result['fixedProductMarketMaker_creationTimestamp'].iloc[0])
            if timestamp > 0:
                return datetime.fromtimestamp(timestamp)
    
    except Exception as e:
        print(f"Error fetching market creation date: {e}")
    
    return None

def get_unique_traders_count(token_ids: List[str]) -> Tuple[int, Dict[str, Any]]:
    """
    Get the count of unique traders and related metrics for the market.
    
    Args:
        token_ids (list): List of token IDs
        
    Returns:
        Tuple[int, Dict[str, Any]]: Number of unique traders and trader metrics
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
    
    all_traders = set()
    trader_metrics = {
        'unique_traders': 0,
        'total_trades': 0,
        'trades_per_token': {},
        'trader_addresses': []
    }
    
    for token_id in token_ids:
        print(f"\nQuerying trader data for token ID: {token_id}")
        
        try:
            # Get all unique traders who have bought or sold this token
            # This uses MarketPosition entity which links users to specific markets
            positions_query = orderbook_subgraph.Query.marketPositions(
                first=1000,  # Adjust as needed for large markets
                where={
                    'market': token_id
                }
            )
            
            # We're interested in the user addresses
            positions_result = sg.query_df([
                positions_query.user.id
            ])
            
            if not positions_result.empty:
                # Get unique traders for this token
                token_traders = set(positions_result['marketPositions_user_id'].unique())
                
                # Store token-specific data
                trader_metrics['trades_per_token'][token_id] = {
                    'trader_count': len(token_traders)
                }
                
                # Add to global set of traders
                all_traders.update(token_traders)
                
                print(f"Found {len(token_traders)} unique traders for this token")
                
                # Also get the first few trader addresses for verification
                if len(trader_metrics['trader_addresses']) < 5:
                    sample_traders = list(token_traders)[:5]
                    for trader in sample_traders:
                        if trader not in trader_metrics['trader_addresses']:
                            trader_metrics['trader_addresses'].append(trader)
            
            # Get additional trade statistics from the orderbook entity
            orderbook_query = orderbook_subgraph.Query.orderbook(
                id=token_id
            )
            
            orderbook_result = sg.query_df([
                orderbook_query.tradesQuantity
            ])
            
            if not orderbook_result.empty and 'orderbook_tradesQuantity' in orderbook_result.columns:
                trades_count = int(orderbook_result['orderbook_tradesQuantity'].iloc[0] or 0)
                trader_metrics['total_trades'] += trades_count
                
                # Save token-specific trade count
                if token_id in trader_metrics['trades_per_token']:
                    trader_metrics['trades_per_token'][token_id]['trades_count'] = trades_count
        
        except Exception as e:
            print(f"Error querying trader data for token {token_id}: {e}")
    
    # Update the final count of unique traders
    trader_count = len(all_traders)
    trader_metrics['unique_traders'] = trader_count
    
    return trader_count, trader_metrics

def extract_enhanced_metrics(market_id: str, token_ids: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract comprehensive metrics for a Polymarket market.
    
    Args:
        market_id (str): Market ID (FixedProductMarketMaker address)
        token_ids (list): List of token IDs
        start_date (str, optional): Market start date in ISO format
        end_date (str, optional): Market end date in ISO format
        
    Returns:
        dict: Enhanced market metrics
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
        'market_id': market_id,
        'token_ids': token_ids,
        'tokens_data': [],
        'creation_date': None,
        'end_date': None,
        'market_duration_days': None,
        'total_trades': 0,
        'total_usdc_volume': 0,
        'avg_daily_volume': None,
        'unique_traders': None,
        'trader_to_volume_ratio': None,
        'trading_frequency': None,
        'buy_sell_ratio': None
    }
    
    # Try to determine market creation date
    if start_date:
        metrics['creation_date'] = start_date
    else:
        creation_date = get_market_creation_date(market_id)
        if creation_date:
            metrics['creation_date'] = creation_date.isoformat()
    
    # Set market end date
    if end_date:
        metrics['end_date'] = end_date
    
    # Calculate market duration if both dates are available
    if metrics['creation_date'] and metrics['end_date']:
        try:
            creation_date = datetime.fromisoformat(metrics['creation_date'].replace('Z', '+00:00'))
            market_end_date = datetime.fromisoformat(metrics['end_date'].replace('Z', '+00:00'))
            
            market_duration = (market_end_date - creation_date).days
            metrics['market_duration_days'] = market_duration
        except Exception as e:
            print(f"Error calculating market duration: {e}")
    
    # Initialize variables for buy/sell aggregation
    total_buy_usdc = 0
    total_sell_usdc = 0
    total_usdc_volume = 0
    
    # Query each token ID for volume and trade data
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
                orderbook_query.sellsQuantity,
                # Volume data
                orderbook_query.scaledCollateralBuyVolume,
                orderbook_query.scaledCollateralSellVolume,
                orderbook_query.scaledCollateralVolume
            ])
            
            if not orderbook_result.empty and not orderbook_result.iloc[0].isnull().all():
                print("Found orderbook data!")
                
                # Extract trade counts
                if 'orderbook_tradesQuantity' in orderbook_result.columns:
                    trades_quantity = int(orderbook_result['orderbook_tradesQuantity'].iloc[0] or 0)
                    token_data['trades_quantity'] = trades_quantity
                    metrics['total_trades'] += trades_quantity
                
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
                
                # Extract volume data
                if 'orderbook_scaledCollateralBuyVolume' in orderbook_result.columns:
                    buy_usdc = float(orderbook_result['orderbook_scaledCollateralBuyVolume'].iloc[0] or 0)
                    token_data['buy_usdc'] = buy_usdc
                    total_buy_usdc += buy_usdc
                
                if 'orderbook_scaledCollateralSellVolume' in orderbook_result.columns:
                    sell_usdc = float(orderbook_result['orderbook_scaledCollateralSellVolume'].iloc[0] or 0)
                    token_data['sell_usdc'] = sell_usdc
                    total_sell_usdc += sell_usdc
                
                if 'orderbook_scaledCollateralVolume' in orderbook_result.columns:
                    volume = float(orderbook_result['orderbook_scaledCollateralVolume'].iloc[0] or 0)
                    token_data['total_usdc'] = volume
                    total_usdc_volume += volume
            else:
                print("No orderbook data found")
        except Exception as e:
            print(f"Error querying orderbook: {e}")
        
        # Add token data to metrics
        metrics['tokens_data'].append(token_data)
    
    # Add total volume metrics
    metrics['total_buy_usdc'] = total_buy_usdc
    metrics['total_sell_usdc'] = total_sell_usdc
    metrics['total_usdc_volume'] = total_usdc_volume
    
    # Calculate buy/sell ratio across all tokens
    if total_sell_usdc > 0:
        metrics['buy_sell_ratio'] = total_buy_usdc / total_sell_usdc
    else:
        metrics['buy_sell_ratio'] = total_buy_usdc if total_buy_usdc > 0 else 0
    
    # Calculate average daily volume if duration is available
    if metrics['market_duration_days'] and metrics['market_duration_days'] > 0:
        metrics['avg_daily_volume'] = total_usdc_volume / metrics['market_duration_days']
    
    # Get unique traders count and related metrics
    try:
        unique_traders, trader_metrics = get_unique_traders_count(token_ids)
        metrics['unique_traders'] = unique_traders
        metrics['trader_metrics'] = trader_metrics
        
        # Calculate trader-to-volume ratio
        if unique_traders > 0:
            metrics['trader_to_volume_ratio'] = total_usdc_volume / unique_traders
        
        # Calculate trading frequency (trades per trader)
        if unique_traders > 0:
            metrics['trading_frequency'] = metrics['total_trades'] / unique_traders
    except Exception as e:
        print(f"Error calculating trader metrics: {e}")
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract enhanced metrics for Polymarket markets")
    parser.add_argument("--market-id", required=True, help="Market ID (FixedProductMarketMaker address)")
    parser.add_argument("--token-ids", required=True, help="Token IDs as JSON string")
    parser.add_argument("--end-date", help="Market end date (e.g., 2024-11-05T12:00:00Z)")
    parser.add_argument("--start-date", help="Market start date (e.g., 2024-01-04T22:58:00Z)")
    parser.add_argument("--output", help="Output file path", default="market_metrics")
    args = parser.parse_args()
    
    try:
        # Parse token IDs
        token_ids = parse_token_ids(args.token_ids)
        if not token_ids:
            print("Error: No valid token IDs found")
            return
        
        print(f"Parsed token IDs: {token_ids}")
        
        # Extract enhanced metrics
        metrics = extract_enhanced_metrics(
            args.market_id,
            token_ids,
            args.start_date,
            args.end_date
        )
        
        # Print results summary
        print("\nMarket Metrics Summary:")
        print("=" * 50)
        print(f"Market ID: {args.market_id}")
        
        # Print key metrics
        if metrics['creation_date']:
            print(f"Creation Date: {metrics['creation_date']}")
        
        if metrics['end_date']:
            print(f"End Date: {metrics['end_date']}")
        
        if metrics['market_duration_days'] is not None:
            print(f"Market Duration: {metrics['market_duration_days']} days")
        
        print(f"Total USDC Volume: ${metrics['total_usdc_volume']:,.2f}")
        
        if metrics['avg_daily_volume'] is not None:
            print(f"Avg Daily Volume: ${metrics['avg_daily_volume']:,.2f}")
        
        print(f"Total Trades: {metrics['total_trades']:,}")
        
        if metrics['unique_traders'] is not None:
            print(f"Unique Traders: {metrics['unique_traders']:,}")
        
        if metrics['trader_to_volume_ratio'] is not None:
            print(f"Volume per Trader: ${metrics['trader_to_volume_ratio']:,.2f}")
        
        if metrics['trading_frequency'] is not None:
            print(f"Trades per Trader: {metrics['trading_frequency']:.2f}")
        
        if metrics['buy_sell_ratio'] is not None:
            print(f"Buy/Sell Ratio: {metrics['buy_sell_ratio']:.2f}")
        
        # Save results to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{args.output}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nDetailed metrics saved to: {output_file}")
        
    except Exception as e:
        print(f"Error executing script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()