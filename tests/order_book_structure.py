#!/usr/bin/env python3
"""
Explore Polymarket orderbook subgraph structure with a simple query.
This helps us understand how to properly query the API.
"""

import os
from dotenv import load_dotenv
from subgrounds import Subgrounds
import pandas as pd
import json

def explore_subgraph():
    """Explore the structure of the orderbook subgraph."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")
    
    # Initialize Subgrounds
    sg = Subgrounds()
    
    # Trump 2024 presidential election market
    question_id = '0xe3b1bc389210504ebcb9cffe4b0ed06ccac50561e0f24abb6379984cec030f00'
    question_id = question_id.lower()
    
    # Connect to orderbook subgraph
    orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    orderbook_subgraph = sg.load_subgraph(orderbook_url)
    
    print("\n1. Basic Market Query")
    print("====================")
    try:
        # Query basic market info
        market_query = orderbook_subgraph.Query.fixedProductMarketMaker(
            id=question_id
        )
        
        result = sg.query_df([
            market_query.id,
            market_query.collateralVolume,
            market_query.scaledCollateralVolume,
            market_query.creationTimestamp,
            market_query.tradesQuantity
        ])
        
        if not result.empty:
            print("Market data found!")
            for col in result.columns:
                print(f"- {col}: {result[col].iloc[0]}")
        else:
            print("No market data found!")
    except Exception as e:
        print(f"Error querying market data: {e}")
    
    print("\n2. Transaction Query Without Timestamp Filter")
    print("===========================================")
    try:
        # Try querying transactions without a timestamp filter first
        transactions_query = orderbook_subgraph.Query.transactions(
            first=5,  # Just get a few to test
            where={
                'market': question_id
            },
            orderBy='timestamp',
            orderDirection='desc'  # Get most recent first
        )
        
        transactions_result = sg.query_df([
            transactions_query.id,
            transactions_query.timestamp,
            transactions_query.tradeAmount,
            transactions_query.type
        ])
        
        if not transactions_result.empty:
            print(f"Found {len(transactions_result)} transactions")
            print("\nMost recent transactions:")
            for i, (_, row) in enumerate(transactions_result.iterrows()):
                print(f"Transaction {i+1}:")
                for col in transactions_result.columns:
                    print(f"- {col}: {row[col]}")
                print()
            
            # For further testing, save the most recent timestamp
            most_recent_ts = int(transactions_result['timestamp'].iloc[0])
            cutoff_ts = most_recent_ts - (48 * 3600)  # 48 hours before most recent
            print(f"Most recent timestamp: {most_recent_ts}")
            print(f"48 hours before: {cutoff_ts}")
        else:
            print("No transactions found!")
    except Exception as e:
        print(f"Error querying transactions: {e}")
    
    print("\n3. Transaction Query With Timestamp Filter")
    print("========================================")
    try:
        # If we got transactions, try with a timestamp filter using the cutoff we calculated
        if 'transactions_result' in locals() and not transactions_result.empty:
            most_recent_ts = int(transactions_result['timestamp'].iloc[0])
            cutoff_ts = most_recent_ts - (48 * 3600)  # 48 hours before most recent
            
            # Query with timestamp filter, using string format
            transactions_query_filtered = orderbook_subgraph.Query.transactions(
                first=5,
                where={
                    'market': question_id,
                    'timestamp_gte': str(cutoff_ts)
                },
                orderBy='timestamp',
                orderDirection='desc'
            )
            
            filtered_result = sg.query_df([
                transactions_query_filtered.id,
                transactions_query_filtered.timestamp,
                transactions_query_filtered.tradeAmount,
                transactions_query_filtered.type
            ])
            
            if not filtered_result.empty:
                print(f"Found {len(filtered_result)} filtered transactions")
                print("First transaction in results:")
                for col in filtered_result.columns:
                    print(f"- {col}: {filtered_result[col].iloc[0]}")
            else:
                print("No filtered transactions found!")
        else:
            print("Skipping timestamp filter test, no transactions available.")
    except Exception as e:
        print(f"Error querying filtered transactions: {e}")
    
    print("\n4. Exploring Schema: Available Fields")
    print("====================================")
    try:
        # Let's see what fields are available for transactions
        print("Checking available fields on Transaction entity...")
        # This is a way to introspect available fields in Subgrounds
        transaction_fields = dir(orderbook_subgraph.Transaction)
        # Filter out internal/dunder methods
        transaction_fields = [f for f in transaction_fields if not f.startswith('_')]
        print(f"Available transaction fields: {transaction_fields}")
        
        # Also check FixedProductMarketMaker fields
        market_fields = dir(orderbook_subgraph.FixedProductMarketMaker)
        market_fields = [f for f in market_fields if not f.startswith('_')]
        print(f"\nAvailable market fields: {market_fields}")
    except Exception as e:
        print(f"Error exploring schema: {e}")
    
if __name__ == "__main__":
    explore_subgraph()