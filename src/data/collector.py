##src/data/collector.py
from typing import Dict, List, Optional
import os
from datetime import datetime
import pandas as pd
from subgrounds import Subgrounds
import logging

class PolymarketDataCollector:
    """
    Comprehensive data collector for Polymarket analysis.
    Integrates data from multiple subgraphs to provide complete market analytics.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sg = Subgrounds()
        
        # Initialize subgraph connections
        self.subgraphs = {
            'activity': self.sg.load_subgraph(
                "https://gateway.thegraph.com/api/{}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp".format(api_key)
            ),
            'orderbook': self.sg.load_subgraph(
                "https://gateway.thegraph.com/api/{}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC".format(api_key)
            ),
            'pnl': self.sg.load_subgraph(
                "https://gateway.thegraph.com/api/{}/subgraphs/id/6c58N5U4MtQE2Y8njfVrrAfRykzfqajMGeTMEvMmskVz".format(api_key)
            )
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_market_trading_metrics(self, market_id: str) -> Dict:
        """
        Fetch comprehensive trading metrics for a specific market.
        
        Args:
            market_id: The unique identifier for the market
            
        Returns:
            Dictionary containing market metrics including volume, trades, prices
        """
        try:
            # Query orderbook data using Orderbook type
            orderbook_data = self.sg.query_df(
                self.subgraphs['orderbook'].Query.orderbook(
                    where={'id': market_id}
                )
            )
            
            # Query market activity using FixedProductMarketMaker type
            activity_data = self.sg.query_df(
                self.subgraphs['activity'].Query.fixedProductMarketMaker(
                    where={'id': market_id}
                )
            )
            
            # Combine metrics
            metrics = {
                'market_id': market_id,
                'total_volume': float(orderbook_data['collateralVolume'].iloc[0]),
                'trade_count': int(orderbook_data['tradesQuantity'].iloc[0]),
                'buy_volume': float(orderbook_data['collateralBuyVolume'].iloc[0]),
                'sell_volume': float(orderbook_data['collateralSellVolume'].iloc[0]),
                'last_price': self._calculate_last_price(market_id),
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error fetching market metrics for {market_id}: {str(e)}")
            return None

    def get_trader_positions(self, market_id: str) -> pd.DataFrame:
        """
        Fetch all trader positions for a given market.
        
        Args:
            market_id: The unique identifier for the market
            
        Returns:
            DataFrame containing trader positions and PnL data
        """
        try:
            positions = self.sg.query_df(
                self.subgraphs['pnl'].Query.userPositions(
                    where={'market': market_id},
                    first=1000
                )
            )
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error fetching trader positions for {market_id}: {str(e)}")
            return pd.DataFrame()

    def get_market_liquidity(self, market_id: str) -> Dict:
        """
        Calculate market liquidity metrics including WBAS.
        
        Args:
            market_id: The unique identifier for the market
            
        Returns:
            Dictionary containing liquidity metrics
        """
        try:
            # Get order book depth
            order_book = self.sg.query_df(
                self.subgraphs['orderbook'].Query.orderBook(
                    id=market_id,
                    first=100,
                    orderBy='timestamp',
                    orderDirection='desc'
                )
            )
            
            # Calculate WBAS and other liquidity metrics
            metrics = {
                'market_id': market_id,
                'wbas': self._calculate_wbas(order_book),
                'depth': len(order_book),
                'last_updated': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity for {market_id}: {str(e)}")
            return None

    def _calculate_wbas(self, order_book: pd.DataFrame) -> float:
        """
        Calculate Weighted Bid-Ask Spread for a market.
        
        Args:
            order_book: DataFrame containing order book data
            
        Returns:
            WBAS value as float
        """
        try:
            # Implementation based on order-book-utils.ts logic
            weighted_bids = (order_book['bid_price'] * order_book['bid_size']).sum()
            weighted_asks = (order_book['ask_price'] * order_book['ask_size']).sum()
            total_volume = order_book['bid_size'].sum() + order_book['ask_size'].sum()
            
            if total_volume == 0:
                return 0.0
                
            return (weighted_asks - weighted_bids) / total_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating WBAS: {str(e)}")
            return 0.0

    def _calculate_last_price(self, market_id: str) -> float:
        """
        Get the last traded price for a market.
        
        Args:
            market_id: The unique identifier for the market
            
        Returns:
            Last traded price as float
        """
        try:
            trades = self.sg.query_df(
                self.subgraphs['orderbook'].Query.orderFilledEvents(
                    where={'market': market_id},
                    first=1,
                    orderBy='timestamp',
                    orderDirection='desc'
                )
            )
            
            if len(trades) > 0:
                return float(trades['takerAmountFilled'].iloc[0]) / float(trades['makerAmountFilled'].iloc[0])
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating last price: {str(e)}")
            return 0.0

    def get_active_markets(self, limit: int = 10) -> List[str]:
        """
        Get a list of active market IDs.
        
        Args:
            limit: Maximum number of markets to return
            
        Returns:
            List of market IDs
        """
        try:
            # Query for recent markets from the FPMM subgraph
            markets = self.sg.query_df(
                self.subgraphs['activity'].Query.fixedProductMarketMakers(
                    first=limit,
                    orderBy='creationTimestamp',
                    orderDirection='desc'
                )
            )
            
            return markets['id'].tolist()
        except Exception as e:
            self.logger.error(f"Error fetching active markets: {str(e)}")
            return []

    def save_market_data(self, market_data: Dict, output_path: str):
        """
        Save collected market data to CSV.
        
        Args:
            market_data: Dictionary containing market metrics
            output_path: Path to save the CSV file
        """
        try:
            df = pd.DataFrame([market_data])
            df.to_csv(output_path, index=False)
            self.logger.info(f"Market data saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving market data: {str(e)}")

# Usage example
if __name__ == "__main__":
    api_key = "3cba9db7fd576b4c4822663ffd805e75"
    collector = PolymarketDataCollector(api_key)
    
    # Example market ID
    market_id = "EXAMPLE_MARKET_ID"
    
    # Collect market data
    market_metrics = collector.get_market_trading_metrics(market_id)
    trader_positions = collector.get_trader_positions(market_id)
    liquidity_metrics = collector.get_market_liquidity(market_id)
    
    # Save data
    collector.save_market_data(market_metrics, "market_metrics.csv")
    trader_positions.to_csv("trader_positions.csv", index=False)