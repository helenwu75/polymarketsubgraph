import os
from dotenv import load_dotenv
import requests

def setup_config(api_key: str = None) -> bool:
    """
    Set up the configuration file and validate the Graph API key.
    
    Args:
        api_key (str, optional): The Graph API key to validate. If None, will prompt for input.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    if not api_key:
        api_key = input("Please enter your Graph API key: ").strip()
    
    # Validate API key format
    if not api_key or len(api_key) < 10:  # Basic validation
        print("Error: API key appears invalid (too short)")
        return False
    
    # Test API key with a simple query to the Polymarket subgraph
    test_url = "https://gateway.thegraph.com/api/"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    try:
        # Simple test query to validate API key
        test_query = {
            'query': '{ _meta { block { number } } }'
        }
        response = requests.post(
            f"{test_url}{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp",
            headers=headers,
            json=test_query
        )
        
        if response.status_code != 200:
            print(f"Error: API key validation failed (Status code: {response.status_code})")
            return False
            
        # Create .env file
        env_content = f"""# The Graph API Configuration
GRAPH_API_KEY={api_key}

# Polymarket Subgraph URLs
SUBGRAPH_URL_PNL=https://gateway.thegraph.com/api/${{GRAPH_API_KEY}}/subgraphs/id/6c58N5U4MtQE2Y8njfVrrAfRykzfqajMGeTMEvMmskVz
SUBGRAPH_URL_ACTIVITY=https://gateway.thegraph.com/api/${{GRAPH_API_KEY}}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp
SUBGRAPH_URL_ORDERBOOK=https://gateway.thegraph.com/api/${{GRAPH_API_KEY}}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC
"""
        
        # Write .env file
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("Configuration file created successfully!")
        print("Testing environment loading...")
        
        # Test loading the environment
        load_dotenv()
        if os.getenv('GRAPH_API_KEY') == api_key:
            print("Environment loading test passed!")
            return True
        else:
            print("Error: Environment loading test failed")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error testing API key: {str(e)}")
        return False

if __name__ == "__main__":
    setup_config()