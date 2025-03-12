#!/usr/bin/env python3
"""
Test script to verify connections to all five Polymarket subgraphs.
"""

import os
import sys
from dotenv import load_dotenv
from subgrounds import Subgrounds
import pandas as pd

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_subgraph_manager

def test_subgraph_connections():
    """Test connections to all five Polymarket subgraphs."""
    print("\n===== Testing Subgraph Connections =====")
    
    try:
        # Get the subgraph manager
        manager = get_subgraph_manager()
        
        # List of all subgraphs to test
        subgraph_names = ['pnl', 'activity', 'orderbook', 'positions', 'open_interest']
        
        # Test each subgraph connection
        for name in subgraph_names:
            print(f"\nTesting {name} subgraph connection...")
            try:
                # Get the subgraph
                subgraph = manager.get_subgraph(name)
                
                if not subgraph:
                    print(f"❌ Error: {name} subgraph not initialized")
                    continue
                
                # Test with a simple query
                result = manager.sg.query_df([
                    subgraph.Query._meta.block.number
                ])
                
                if not result.empty:
                    print(f"✅ Successfully connected to {name} subgraph")
                    print(f"   Current block number: {result.iloc[0, 0]}")
                else:
                    print(f"❌ Query returned empty result for {name} subgraph")
            
            except Exception as e:
                print(f"❌ Error testing {name} subgraph: {str(e)}")
        
        print("\n===== Basic Entity Count Test =====")
        # For each subgraph, attempt to query a basic entity count
        query_tests = {
            'pnl': ('userPositions', 5),
            'activity': ('conditions', 5),
            'orderbook': ('markets', 5),
            'positions': ('positions', 5),
            'open_interest': ('markets', 5)
        }
        
        for name, (entity, limit) in query_tests.items():
            try:
                subgraph = manager.get_subgraph(name)
                if not subgraph:
                    print(f"❌ {name} subgraph not available for entity count test")
                    continue
                
                # Generic query to count entities
                query = getattr(subgraph.Query, entity)(first=limit)
                query_path = [query]
                
                # Try to add an ID field if available
                try:
                    query_path = [query.id]
                except:
                    # If id is not available, just use the basic query
                    pass
                
                # Execute the query
                result = manager.sg.query_df(query_path)
                
                if not result.empty:
                    print(f"✅ {name} subgraph: Found {len(result)} {entity}")
                else:
                    print(f"⚠️ {name} subgraph: No {entity} found (may be normal)")
            
            except Exception as e:
                print(f"❌ Error querying {entity} from {name} subgraph: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error initializing subgraph manager: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Test all subgraph connections
    test_subgraph_connections()