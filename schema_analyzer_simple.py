#!/usr/bin/env python3
import os
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import requests

@dataclass
class SubgraphInfo:
    name: str
    url: str
    description: str

class SchemaAnalyzer:
    def __init__(self, api_key: str):
        print("Initializing SchemaAnalyzer...")  # Debug print
        self.api_key = api_key
        self.logger = self._setup_logger()
        self._initialize_subgraphs()
        self._validate_config()
        
    def _initialize_subgraphs(self):
        print("Setting up subgraphs...")  # Debug print
        self.subgraphs = {
            'pnl': SubgraphInfo(
                'PNL Subgraph',
                os.getenv('SUBGRAPH_URL_PNL', ''),
                'Profit and Loss Data'
            ),
            'activity': SubgraphInfo(
                'Activity Subgraph',
                os.getenv('SUBGRAPH_URL_ACTIVITY', ''),
                'Market Activity Tracking'
            ),
            'orderbook': SubgraphInfo(
                'Orderbook Subgraph',
                os.getenv('SUBGRAPH_URL_ORDERBOOK', ''),
                'Order Book and Pricing Information'
            )
        }
        print(f"Subgraphs initialized: {list(self.subgraphs.keys())}")  # Debug print

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('SchemaAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _validate_config(self):
        print("Validating configuration...")  # Debug print
        if not self.api_key:
            raise ValueError("API key is not set")
        
        for subgraph_id, info in self.subgraphs.items():
            if not info.url:
                raise ValueError(f"URL for {subgraph_id} subgraph is not set")
            print(f"Validated {subgraph_id} URL: {info.url[:30]}...")  # Debug print

    def test_connections(self):
        for subgraph_id, info in self.subgraphs.items():
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
                query = "{ __schema { queryType { name } } }"
                response = requests.post(
                    info.url,
                    headers=headers,
                    json={'query': query}
                )
                if response.status_code == 200:
                    print(f"✅ Successfully connected to {subgraph_id}")
                else:
                    print(f"❌ Failed to connect to {subgraph_id}: {response.status_code}")
            except Exception as e:
                print(f"❌ Error connecting to {subgraph_id}: {str(e)}")

def main():
    print("Starting schema analysis...")
    load_dotenv()
    
    api_key = os.getenv('GRAPH_API_KEY')
    if not api_key:
        print("Error: GRAPH_API_KEY environment variable not set")
        return
    
    print(f"Loaded API key: {api_key[:6]}...{api_key[-4:]}")
    
    try:
        analyzer = SchemaAnalyzer(api_key)
        analyzer.test_connections()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()