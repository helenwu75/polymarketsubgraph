import os
import sys
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data import PolymarketDataCollector

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize collector
    api_key = os.getenv('GRAPH_API_KEY', '3cba9db7fd576b4c4822663ffd805e75')
    collector = PolymarketDataCollector(api_key)
    
    try:
        # Get active markets
        active_markets = collector.get_active_markets(limit=5)
        if not active_markets:
            print("No active markets found")
            return
            
        print(f"Found {len(active_markets)} active markets")
        
        # Process each market
        for market_id in active_markets:
            print(f"\nProcessing market {market_id}")
            
            # Collect data
            market_metrics = collector.get_market_trading_metrics(market_id)
            if market_metrics:
                print("Successfully collected market metrics:")
                print(market_metrics)
                
                # Save to data directory
                output_dir = os.path.join(project_root, "data", "raw")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"market_{market_id}_metrics.csv")
                collector.save_market_data(market_metrics, output_path)
                print(f"Saved data to {output_path}")
            else:
                print(f"Failed to collect metrics for market {market_id}")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()