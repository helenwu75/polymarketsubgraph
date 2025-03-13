import os
from dotenv import load_dotenv
import requests

def test_configuration():
    """
    Test the configuration setup by making a simple query to each subgraph.
    """
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('GRAPH_API_KEY')
    if not api_key:
        print("Error: GRAPH_API_KEY not found in environment")
        return False
    
    # Test each subgraph endpoint
    subgraphs = {
        'PNL': os.getenv('SUBGRAPH_URL_PNL'),
        'Activity': os.getenv('SUBGRAPH_URL_ACTIVITY'),
        'Orderbook': os.getenv('SUBGRAPH_URL_ORDERBOOK'),
        'Positions': os.getenv('SUBGRAPH_URL_POSITIONS'),
        'Open Interest': os.getenv('SUBGRAPH_URL_OPEN_INTEREST')
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    test_query = {
        'query': '{ _meta { block { number } } }'
    }
    
    for name, url in subgraphs.items():
        if not url:
            print(f"Error: URL for {name} subgraph not found in environment")
            continue
            
        try:
            print(f"\nTesting {name} subgraph...")
            response = requests.post(url, headers=headers, json=test_query)
            
            if response.status_code == 200:
                block_number = response.json().get('data', {}).get('_meta', {}).get('block', {}).get('number')
                print(f"✓ {name} subgraph is accessible (Latest block: {block_number})")
            else:
                print(f"✗ {name} subgraph test failed (Status code: {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Error testing {name} subgraph: {str(e)}")

if __name__ == "__main__":
    print("Testing The Graph API configuration...")
    test_configuration()