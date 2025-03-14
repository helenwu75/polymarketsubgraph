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
import contextlib

# Constants for pagination and timeouts
BATCH_SIZE = 1000
REQUEST_TIMEOUT = 120  # Seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds
MAX_TRADERS_PER_TOKEN = 1000  # Limit to prevent excessive runtime

# ConfigSettings class to allow runtime modification of constants
class ConfigSettings:
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.request_timeout = REQUEST_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        self.max_traders_per_token = MAX_TRADERS_PER_TOKEN

# Global configuration object
config = ConfigSettings()

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

def create_subgrounds_session(timeout=None):
    """
    Create a Subgrounds instance with proper timeout settings.
    
    Args:
        timeout (int): Request timeout in seconds
        
    Returns:
        Subgrounds: Configured Subgrounds instance
    """
    sg = Subgrounds()
    
    # Use global config if timeout is not specified
    if timeout is None:
        timeout = config.request_timeout
    
    # Configure timeout if the client allows it
    if hasattr(sg, '_session') and hasattr(sg._session, 'request'):
        sg._session.request_kwargs = {"timeout": timeout}
    
    return sg

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
    
    # Initialize Subgrounds with timeout
    sg = create_subgrounds_session()
    
    try:
        # Connect to activity subgraph (which has creation timestamps)
        activity_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
        activity_subgraph = sg.load_subgraph(activity_url)
        
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
    
    finally:
        # Clean up connections
        close_subgrounds_session(sg)
    
    return None

# This function is no longer used, as the pagination logic has been moved directly into get_unique_traders_count
# Keeping the function signature for reference in case it's needed elsewhere
def get_traders_with_pagination(subgraph, token_id, max_traders=MAX_TRADERS_PER_TOKEN):
    """
    Get unique traders for a token with pagination to handle large datasets.
    This function is deprecated and no longer used.
    
    Args:
        subgraph: Subgrounds subgraph instance
        token_id (str): Token ID to query
        max_traders (int): Maximum number of traders to retrieve
        
    Returns:
        set: Set of unique trader addresses
    """
    print("Warning: This function is deprecated. Use the pagination in get_unique_traders_count instead.")
    
    # Return an empty set to avoid confusion
    return set()

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
    
    # Initialize Subgrounds with timeout
    sg = create_subgrounds_session()
    
    all_traders = set()
    trader_metrics = {
        'unique_traders': 0,
        'total_trades': 0,
        'trades_per_token': {},
        'trader_addresses': []
    }
    
    try:
        # Connect to orderbook subgraph
        orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
        orderbook_subgraph = sg.load_subgraph(orderbook_url)
        
        for token_id in token_ids:
            print(f"\nQuerying trader data for token ID: {token_id}")
            
            try:
                # Initialize for pagination
                token_traders = set()
                skip = 0
                
                print(f"Retrieving traders for token {token_id} (paginated)...")
                
                while True:
                    try:
                        # Query the next batch of positions
                        positions_query = orderbook_subgraph.Query.marketPositions(
                            first=config.batch_size,
                            skip=skip,
                            where={
                                'market': token_id
                            }
                        )
                        
                        positions_result = sg.query_df([
                            positions_query.user.id
                        ])
                        
                        # Check if we got any results
                        if positions_result.empty:
                            print(f"  No more traders found after {skip} records")
                            break
                            
                        # Get unique traders from this batch
                        batch_traders = set(positions_result['marketPositions_user_id'].unique())
                        batch_size = len(batch_traders)
                        
                        # Add to the token-specific set
                        token_traders.update(batch_traders)
                        
                        print(f"  Retrieved {batch_size} traders (batch), {len(token_traders)} unique traders (total)")
                        
                        # Stop if we've reached the max traders limit
                        if len(token_traders) >= config.max_traders_per_token:
                            print(f"  Reached maximum trader limit of {config.max_traders_per_token}")
                            break
                            
                        # Stop if this batch was smaller than the requested size
                        if batch_size < config.batch_size:
                            break
                            
                        # Move to the next batch
                        skip += config.batch_size
                        
                        # Brief pause to avoid overwhelming the API
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"  Error during trader pagination: {e}")
                        # Try one more time with a delay
                        time.sleep(config.retry_delay)
                        try:
                            positions_query = orderbook_subgraph.Query.marketPositions(
                                first=config.batch_size,
                                skip=skip,
                                where={
                                    'market': token_id
                                }
                            )
                            
                            positions_result = sg.query_df([
                                positions_query.user.id
                            ])
                            
                            if not positions_result.empty:
                                batch_traders = set(positions_result['marketPositions_user_id'].unique())
                                token_traders.update(batch_traders)
                                print(f"  Retry succeeded, got {len(batch_traders)} more traders")
                                skip += config.batch_size
                                continue
                        except:
                            pass
                        # Continue with what we have so far if retry fails
                        break
                
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
                for retry in range(config.max_retries):
                    try:
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
                        
                        # If successful, break retry loop
                        break
                        
                    except Exception as e:
                        if retry < config.max_retries - 1:
                            delay = config.retry_delay * (2 ** retry)  # Exponential backoff
                            print(f"Error querying trade statistics (attempt {retry+1}/{config.max_retries}). Retrying in {delay}s: {e}")
                            time.sleep(delay)
                        else:
                            print(f"Failed to get trade statistics after {config.max_retries} attempts: {e}")
            
            except Exception as e:
                print(f"Error querying trader data for token {token_id}: {e}")
    
    finally:
        # Clean up connections
        close_subgrounds_session(sg)
    
    # Update the final count of unique traders
    trader_count = len(all_traders)
    trader_metrics['unique_traders'] = trader_count
    
    return trader_count, trader_metrics

def close_subgrounds_session(sg):
    """
    Properly close a Subgrounds session to avoid resource warnings.
    
    Args:
        sg: Subgrounds instance to close
    """
    if hasattr(sg, '_session'):
        with contextlib.suppress(Exception):
            if hasattr(sg._session, 'close'):
                sg._session.close()
            
        # Explicitly close any underlying requests session as well
        if hasattr(sg._session, '_session'):
            with contextlib.suppress(Exception):
                sg._session._session.close()
        
        # If there's a transport adapter with connections, close those too
        if hasattr(sg._session, '_session') and hasattr(sg._session._session, 'adapters'):
            for adapter in sg._session._session.adapters.values():
                with contextlib.suppress(Exception):
                    adapter.close()

def retry_with_backoff(func, max_retries=None, initial_delay=None):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        The result of the function or raises the last exception
    """
    # Use global config if parameters are not specified
    if max_retries is None:
        max_retries = config.max_retries
    if initial_delay is None:
        initial_delay = config.retry_delay
        
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                raise

def extract_enhanced_metrics(market_id: str, token_ids: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, market_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract comprehensive metrics for a Polymarket market.
    
    Args:
        market_id (str): Market ID (FixedProductMarketMaker address)
        token_ids (list): List of token IDs
        start_date (str, optional): Market start date in ISO format
        end_date (str, optional): Market end date in ISO format
        market_name (str, optional): Market name or question for the filename
        
    Returns:
        dict: Enhanced market metrics
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")
    
    # Initialize Subgrounds with timeout
    sg = create_subgrounds_session()
    
    # Initialize metrics dictionary
    metrics = {
        'market_id': market_id,
        'market_name': market_name,
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
    
    try:
        # Connect to orderbook subgraph
        orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
        orderbook_subgraph = sg.load_subgraph(orderbook_url)
        
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
            
            # Get basic orderbook stats with retry logic
            def query_orderbook():
                orderbook_query = orderbook_subgraph.Query.orderbook(
                    id=token_id
                )
                
                return sg.query_df([
                    orderbook_query.id,
                    orderbook_query.tradesQuantity,
                    orderbook_query.buysQuantity,
                    orderbook_query.sellsQuantity,
                    # Volume data
                    orderbook_query.scaledCollateralBuyVolume,
                    orderbook_query.scaledCollateralSellVolume,
                    orderbook_query.scaledCollateralVolume
                ])
            
            try:
                orderbook_result = retry_with_backoff(query_orderbook)
                
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
    
    finally:
        # Clean up connections
        close_subgrounds_session(sg)
    
    # Get unique traders count and related metrics
    # This uses a separate Subgrounds session to prevent connection issues
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
    parser.add_argument("--market-id", help="Market ID (FixedProductMarketMaker address)")
    parser.add_argument("--token-ids", help="Token IDs as JSON string")
    parser.add_argument("--end-date", help="Market end date (e.g., 2024-11-05T12:00:00Z)")
    parser.add_argument("--start-date", help="Market start date (e.g., 2024-01-04T22:58:00Z)")
    parser.add_argument("--market-name", help="Market name (for filename)")
    parser.add_argument("--output-dir", help="Output directory for files", default="market_metrics")
    parser.add_argument("--max-traders", type=int, help=f"Maximum traders to retrieve per token (default: {config.max_traders_per_token})")
    parser.add_argument("--timeout", type=int, help=f"Request timeout in seconds (default: {config.request_timeout})")
    args = parser.parse_args()
    
    # Update configuration if provided via arguments
    if args.max_traders:
        config.max_traders_per_token = args.max_traders
    if args.timeout:
        config.request_timeout = args.timeout
    
    try:
        # Set default values for Trump 2024 election market if not provided
        if not args.market_id:
            args.market_id = "0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917"
            print(f"Using default market ID: {args.market_id}")
            
        if not args.token_ids:
            args.token_ids = '["21742633143463906290569050155826241533067272736897614950488156847949938836455", "48331043336612883890938759509493159234755048973500640148014422747788308965732"]'
            print(f"Using default token IDs for Trump 2024 election market")
            
        if not args.start_date:
            args.start_date = "2024-01-04T22:58:00Z"
            print(f"Using default start date: {args.start_date}")
            
        if not args.end_date:
            args.end_date = "2024-11-05T12:00:00Z"
            print(f"Using default end date: {args.end_date}")
            
        if not args.market_name:
            args.market_name = "Trump_2024_Presidential_Election"
            print(f"Using default market name: {args.market_name}")
        
        # Parse token IDs
        token_ids = parse_token_ids(args.token_ids)
        if not token_ids:
            print("Error: No valid token IDs found")
            return
        
        print(f"Parsed token IDs: {token_ids}")
        print(f"Using request timeout: {config.request_timeout}s")
        print(f"Maximum traders per token: {config.max_traders_per_token}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving metrics to directory: {args.output_dir}")
        
        # Extract enhanced metrics
        metrics = extract_enhanced_metrics(
            args.market_id,
            token_ids,
            args.start_date,
            args.end_date,
            args.market_name
        )
        
        # Print results summary
        print("\nMarket Metrics Summary:")
        print("=" * 50)
        print(f"Market ID: {args.market_id}")
        print(f"Market Name: {args.market_name}")
        
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
        
        # Generate sanitized market name for filename
        sanitized_name = args.market_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        sanitized_name = ''.join(c for c in sanitized_name if c.isalnum() or c in ['_', '-'])
        
        # Save results to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{args.output_dir}/{sanitized_name}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nDetailed metrics saved to: {output_file}")
        
    except Exception as e:
        print(f"Error executing script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()