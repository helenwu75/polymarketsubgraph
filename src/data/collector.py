from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketQuery:
    """Structure for querying market data with validation"""
    market_id: str
    include_trading_history: bool = False
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None

class PolymarketDataCollector:
    def __init__(self):
        """Initialize collector with proper error handling"""
        try:
            # Load environment
            load_dotenv()
            self.api_key = os.getenv('GRAPH_API_KEY')
            if not self.api_key:
                raise ValueError("GRAPH_API_KEY not found in environment variables")

            # Initialize Subgrounds
            self.sg = Subgrounds()
            
            # Initialize subgraphs with proper error handling
            self.subgraphs = self._initialize_subgraphs()
            
        except Exception as e:
            logger.error(f"Failed to initialize collector: {str(e)}")
            raise

    def _initialize_subgraphs(self) -> Dict:
        """Initialize all required subgraph connections"""
        try:
            return {
                'activity': self.sg.load_subgraph(
                    f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
                ),
                'orderbook': self.sg.load_subgraph(
                    f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
                ),
                'pnl': self.sg.load_subgraph(
                    f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/6c58N5U4MtQE2Y8njfVrrAfRykzfqajMGeTMEvMmskVz"
                )
            }
        except Exception as e:
            logger.error(f"Failed to initialize subgraphs: {str(e)}")
            raise

    def validate_market_id(self, market_id: str) -> bool:
        """
        Validate market ID format
        
        Args:
            market_id: Market ID to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(market_id, str):
            return False
        
        # Check if it's a valid Ethereum address format
        if not market_id.startswith('0x') or len(market_id) != 42:
            return False
            
        try:
            # Verify it's valid hex
            int(market_id[2:], 16)
            return True
        except ValueError:
            return False

    async def get_basic_market_info(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basic market information using the Activity subgraph
        
        Args:
            market_id: Market ID to query
            
        Returns:
            Dictionary containing basic market information or None if error
        """
        try:
            if not self.validate_market_id(market_id):
                raise ValueError(f"Invalid market ID format: {market_id}")

            # Using fields from FixedProductMarketMaker schema
            query = self.subgraphs['activity'].Query.fixedProductMarketMaker(
                id=market_id,
                selection={
                    'id': True,
                    'creationTimestamp': True,
                    'creationTransactionHash': True,
                    'creator': True,
                    'fee': True,
                    'collateralToken': {
                        'id': True,
                        'decimals': True,
                        'symbol': True
                    },
                    'conditions': {
                        'id': True,
                        'outcomeSlotCount': True
                    },
                    'lastActiveDay': True,
                    'totalSupply': True
                }
            )

            result = self.sg.query_df([query])
            
            if result.empty:
                logger.warning(f"No data found for market {market_id}")
                return None

            # Transform to more usable format
            market_data = {
                'market_id': market_id,
                'creation_date': datetime.fromtimestamp(int(result['creationTimestamp'].iloc[0])),
                'creator': result['creator'].iloc[0],
                'fee_percentage': float(result['fee'].iloc[0]) / 1e18,  # Convert from base units
                'collateral_token': result['collateralToken_symbol'].iloc[0],
                'collateral_decimals': int(result['collateralToken_decimals'].iloc[0]),
                'last_active': datetime.fromtimestamp(int(result['lastActiveDay'].iloc[0])),
                'total_supply': int(result['totalSupply'].iloc[0])
            }
            
            return market_data

        except Exception as e:
            logger.error(f"Error fetching basic market info for {market_id}: {str(e)}")
            return None