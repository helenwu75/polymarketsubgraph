# src/config/example_usage.py

from src.config.settings import get_subgraph_manager

def test_subgraph_setup():
    """Test subgraph configuration and basic queries."""
    
    # Use context manager for proper resource cleanup
    with get_subgraph_manager() as manager:
        # Test each subgraph
        for subgraph_name in ['pnl', 'activity', 'orderbook']:
            print(f"\nTesting {subgraph_name} subgraph:")
            
            # Get the subgraph
            subgraph = manager.get_subgraph(subgraph_name)
            
            # Verify connection
            if manager.verify_connection(subgraph_name):
                print(f"✓ Connection verified")
                
                # Get latest block number
                result = manager.sg.query_df([
                    subgraph.Query._meta.block.number
                ])
                print(f"Latest block number: {result.iloc[0,0]}")
            else:
                print(f"✗ Connection failed")

if __name__ == "__main__":
    test_subgraph_setup()