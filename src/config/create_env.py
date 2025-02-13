from pathlib import Path

def create_env_file():
    """Create .env file with API key"""
    env_content = """# The Graph API Configuration
GRAPH_API_KEY=3cba9db7fd576b4c4822663ffd805e75

# Polymarket Subgraph URLs
SUBGRAPH_URL_PNL=https://gateway.thegraph.com/api/${GRAPH_API_KEY}/subgraphs/id/6c58N5U4MtQE2Y8njfVrrAfRykzfqajMGeTMEvMmskVz
SUBGRAPH_URL_ACTIVITY=https://gateway.thegraph.com/api/${GRAPH_API_KEY}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp
SUBGRAPH_URL_ORDERBOOK=https://gateway.thegraph.com/api/${GRAPH_API_KEY}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC
"""
    
    # Create .env file
    env_path = Path('.env')
    env_path.write_text(env_content)
    print("Created .env file with API key")

if __name__ == "__main__":
    create_env_file()