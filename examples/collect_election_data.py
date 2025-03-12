import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.election_collector import ElectionMarketCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_market_summary(markets_df, stats, thresholds):
    """Print a summary of the market analysis"""
    print("\nMarket Analysis Summary")
    print("=" * 50)
    
    print(f"\nTotal Markets Analyzed: {len(markets_df)}")
    
    if 'volume_stats' in stats:
        print("\nVolume Statistics:")
        print(f"Total Volume: ${stats['volume_stats'].get('total_volume', 0):,.2f}")
        print(f"Mean Volume: ${stats['volume_stats'].get('mean_volume', 0):,.2f}")
        print(f"Median Volume: ${stats['volume_stats'].get('median_volume', 0):,.2f}")
    
    if 'trading_stats' in stats:
        print("\nTrading Statistics:")
        print(f"Total Trades: {stats['trading_stats'].get('total_trades', 0):,}")
        print(f"Mean Trades: {stats['trading_stats'].get('mean_trades', 0):.2f}")
        print(f"Median Trades: {stats['trading_stats'].get('median_trades', 0)}")
    
    if thresholds:
        print("\nSuggested Thresholds:")
        for metric, values in thresholds.items():
            print(f"\n{metric.capitalize()}:")
            for level, value in values.items():
                print(f"- {level}: {value:,.2f}")

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")
    
    try:
        # Initialize collector
        collector = ElectionMarketCollector(api_key)
        
        # Fetch election market data
        logger.info("Fetching election market data...")
        markets, stats = collector.get_election_markets()
        
        if markets.empty:
            logger.warning("No market data collected")
            return
        
        # Get suggested thresholds
        thresholds = collector.suggest_market_thresholds(stats)
        
        # Print summary
        print_market_summary(markets, stats, thresholds)
        
        # Save summary to file
        with open('data/raw/market_summary.txt', 'w') as f:
            f.write("Market Analysis Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"\nTotal Markets: {len(markets)}\n")
            f.write(f"Total Volume: ${stats['volume_stats'].get('total_volume', 0):,.2f}\n")
            f.write(f"Total Trades: {stats['trading_stats'].get('total_trades', 0):,}\n")
        
        logger.info("Analysis complete. Data saved to data/raw/")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()