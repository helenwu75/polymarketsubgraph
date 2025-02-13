# test_connection.py

from subgrounds import Subgrounds
import os
from dotenv import load_dotenv

def test_subgraph_connection():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("API key not found in .env file")
    
    # Initialize Subgrounds
    sg = Subgrounds()
    
    # Define subgraph URLs
    subgraphs = {
        'pnl': f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/6c58N5U4MtQE2Y8njfVrrAfRykzfqajMGeTMEvMmskVz",
        'activity': f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp",
        'orderbook': f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    }
    
    # Test each subgraph connection
    for name, url in subgraphs.items():
        try:
            print(f"\nTesting connection to {name} subgraph...")
            subgraph = sg.load_subgraph(url)
            
            # Try a simple query to verify connection
            result = sg.query_df([
                subgraph.Query._meta.block.number
            ])
            
            print(f"✓ Successfully connected to {name} subgraph")
            print(f"Latest block number: {result.iloc[0,0]}")
            
        except Exception as e:
            print(f"✗ Failed to connect to {name} subgraph")
            print(f"Error: {str(e)}")
            raise

if __name__ == "__main__":
    test_subgraph_connection()