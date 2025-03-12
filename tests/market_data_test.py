import asyncio
from subgrounds import Subgrounds
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_market_query():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    if not api_key:
        raise ValueError("GRAPH_API_KEY not found in environment variables")

    # Initialize Subgrounds
    sg = Subgrounds()
    
    try:
        # Connect to activity subgraph
        activity_subgraph = sg.load_subgraph(
            f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
        )
        
        logger.info("Querying markets...")
        
        # Most basic query possible
        markets_data = sg.query_df([
            activity_subgraph.Query.fixedProductMarketMakers(
                first=10
            )
        ])
        
        if not markets_data.empty:
            logger.info("\nMarkets found!")
            logger.info(f"\nColumns available: {markets_data.columns.tolist()}")
            
            for idx, row in markets_data.iterrows():
                logger.info("\n-------------------")
                logger.info(f"Row {idx + 1}:")
                for col in markets_data.columns:
                    value = row[col]
                    if value is not None and str(value).strip() != '':
                        logger.info(f"{col}: {value}")
            
            return markets_data
        else:
            logger.info("No markets found")
            return None
            
    except Exception as e:
        logger.error(f"Error querying market: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    markets_df = asyncio.run(test_market_query())
    if markets_df is not None:
        print("\nRetrieved market data. Check the logs for details.")
        print(f"\nTotal markets found: {len(markets_df)}")
    else:
        print("\nNo market data retrieved. Check the error logs.")