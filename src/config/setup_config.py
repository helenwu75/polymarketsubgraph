import os
import contextlib
from typing import Dict, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from subgrounds import Subgrounds
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SubgraphConfig:
    """Configuration class for subgraph endpoints."""
    pnl_url: str
    activity_url: str
    orderbook_url: str
    api_key: str

class SubgraphManager:
    """Manages subgraph connections and configurations."""
    
    def __init__(self):
        self.config = self._load_config()
        self.sg = Subgrounds()
        self._subgraphs: Dict[str, any] = {}
        
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'sg'):
            # Close any open sessions
            with contextlib.suppress(Exception):
                if hasattr(self.sg, '_session') and self.sg._session:
                    self.sg._session.close()
    
    def _load_config(self) -> SubgraphConfig:
        """Load configuration from environment variables."""
        load_dotenv()
        
        api_key = os.getenv('GRAPH_API_KEY')
        if not api_key:
            raise ValueError("GRAPH_API_KEY not found in environment variables")
            
        return SubgraphConfig(
            pnl_url=f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/6c58N5U4MtQE2Y8njfVrrAfRykzfqajMGeTMEvMmskVz",
            activity_url=f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp",
            orderbook_url=f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC",
            api_key=api_key
        )
    
    def _load_subgraph(self, name: str, url: str) -> None:
        """Load a specific subgraph."""
        try:
            logger.info(f"Loading {name} subgraph...")
            self._subgraphs[name] = self.sg.load_subgraph(url)
            logger.info(f"Successfully loaded {name} subgraph")
        except Exception as e:
            logger.error(f"Error loading {name} subgraph: {str(e)}")
            raise
    
    def initialize_subgraphs(self) -> None:
        """Initialize all required subgraphs."""
        self._load_subgraph('pnl', self.config.pnl_url)
        self._load_subgraph('activity', self.config.activity_url)
        self._load_subgraph('orderbook', self.config.orderbook_url)
    
    def get_subgraph(self, name: str) -> Optional[any]:
        """Get a specific subgraph instance."""
        return self._subgraphs.get(name)
    
    def verify_connection(self, name: str) -> bool:
        """Verify connection to a specific subgraph."""
        try:
            subgraph = self.get_subgraph(name)
            if not subgraph:
                return False
                
            # Try a simple query to verify connection
            result = self.sg.query_df([
                subgraph.Query._meta.block.number
            ])
            return not result.empty
        except Exception as e:
            logger.error(f"Connection verification failed for {name}: {str(e)}")
            return False

def get_subgraph_manager() -> SubgraphManager:
    """Get an initialized SubgraphManager instance."""
    warnings.filterwarnings('ignore', category=ResourceWarning)
    manager = SubgraphManager()
    manager.initialize_subgraphs()
    return manager