#!/usr/bin/env python3
"""
Token Diagnostics Script

Tests different query approaches to diagnose why certain tokens aren't returning trade data.
"""

import os
import json
import time
import argparse
import requests
from dotenv import load_dotenv

def test_token_query(token_id, api_key, query_type="enrichedOrderFilleds"):
    """Test different query approaches for a token."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    queries = {
        # Standard query for trades
        "enrichedOrderFilleds": f"""
        {{
          enrichedOrderFilleds(
            first: 5
            where: {{ market: "{token_id}" }}
          ) {{
            id
            timestamp
            price
          }}
        }}
        """,
        
        # Check if token exists in orderbook
        "orderbook": f"""
        {{
          orderbook(id: "{token_id}") {{
            id
            tradesQuantity
            collateralVolume
          }}
        }}
        """,
        
        # Check market data
        "marketData": f"""
        {{
          marketData(id: "{token_id}") {{
            id
            outcomeIndex
            priceOrderbook
          }}
        }}
        """,
        
        # Try exact and case-insensitive match
        "altFormatQuery": f"""
        {{
          orderbooks(
            where: {{ id_in: ["{token_id}", "{token_id.lower()}", "{token_id.strip()}", "{token_id.upper()}"] }}
          ) {{
            id
            tradesQuantity
          }}
        }}
        """
    }
    
    # If only testing one query type
    if query_type in queries:
        query_str = queries[query_type]
        response = requests.post(
            url, 
            headers=headers, 
            json={'query': query_str},
            timeout=30
        )
        return response.json()
    
    # Test all queries
    results = {}
    for name, query_str in queries.items():
        print(f"Testing query: {name}")
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json={'query': query_str},
                timeout=30
            )
            results[name] = response.json()
            time.sleep(1)  # Avoid rate limiting
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

def test_similar_tokens(token_id, api_key):
    """Test variations of the token ID to check for formatting issues."""
    variations = [
        token_id,
        token_id.lower(),
        token_id.strip(),
        # Try removing leading zeros
        token_id.lstrip('0'),
        # Try with different number of digits
        token_id[:len(token_id)-1],
        token_id[:len(token_id)-2],
        # Try as integer (no quotes)
        f"{{id: {token_id}}}"
    ]
    
    results = {}
    for i, variation in enumerate(variations):
        print(f"Testing variation {i+1}: {variation}")
        try:
            url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            query = f"""
            {{
              enrichedOrderFilleds(
                first: 1
                where: {{ market: "{variation}" }}
              ) {{
                id
              }}
            }}
            """
            
            response = requests.post(
                url, 
                headers=headers, 
                json={'query': query},
                timeout=30
            )
            results[f"variation_{i+1}"] = response.json()
            time.sleep(1)  # Avoid rate limiting
        except Exception as e:
            results[f"variation_{i+1}"] = {'error': str(e)}
    
    return results

def test_subgraph_health(api_key):
    """Test the overall health of the subgraph."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Test a simple query to check overall responsiveness
    query = """
    {
      _meta {
        block {
          number
        }
        deployment
        hasIndexingErrors
      }
      globals(first: 1) {
        id
        tradesQuantity
      }
    }
    """
    
    try:
        response = requests.post(
            url, 
            headers=headers, 
            json={'query': query},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {'error': str(e)}

def test_alternative_approaches(token_id, api_key):
    """Test alternative approaches like querying by transaction hash or related entities."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Try to find any trades via market positions
    query = f"""
    {{
      marketPositions(
        first: 5
        where: {{ market: "{token_id}" }}
      ) {{
        id
        user {{
          id
        }}
        netQuantity
        netValue
      }}
    }}
    """
    
    try:
        response = requests.post(
            url, 
            headers=headers, 
            json={'query': query},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {'error': str(e)}

def test_working_token(api_key):
    """Test a token that is known to work to compare responses."""
    # Use the first token with trades from the original data
    token_id = "86880762101401386532499499551168760751416168057555342511375236617186871984326"
    return test_token_query(token_id, api_key)

def main():
    parser = argparse.ArgumentParser(description="Token Query Diagnostics")
    parser.add_argument("--token", required=True, default="98660164847018785053515195952319232386051532199528964628112014110011016359177", help="Token ID to diagnose")
    parser.add_argument("--all-tests", action="store_true", help="Run all diagnostic tests")
    parser.add_argument("--output", default="reports/token_diagnostics.json", help="Output file")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GRAPH_API_KEY')
    
    if not api_key:
        print("ERROR: GRAPH_API_KEY not found in environment variables")
        return
    
    print(f"Running diagnostics on token: {args.token}")
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Run basic tests
    results = {
        "token_id": args.token,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "basic_query": test_token_query(args.token, api_key, "enrichedOrderFilleds"),
        "orderbook_query": test_token_query(args.token, api_key, "orderbook"),
        "marketData_query": test_token_query(args.token, api_key, "marketData")
    }
    
    # Run additional tests if requested
    if args.all_tests:
        print("Running advanced tests...")
        results["variations_test"] = test_similar_tokens(args.token, api_key)
        results["subgraph_health"] = test_subgraph_health(api_key)
        results["alternative_approaches"] = test_alternative_approaches(args.token, api_key)
        results["working_token_response"] = test_working_token(api_key)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Diagnostics saved to {args.output}")
    
    # Display key findings
    print("\nKEY FINDINGS:")
    
    # Check if we got any data back for the token
    has_trades = False
    has_orderbook = False
    has_market_data = False
    
    if results["basic_query"].get("data", {}).get("enrichedOrderFilleds"):
        has_trades = True
        print("✅ Found trades for this token")
    else:
        print("❌ No trades found")
    
    if results["orderbook_query"].get("data", {}).get("orderbook"):
        has_orderbook = True
        orderbook = results["orderbook_query"]["data"]["orderbook"]
        print(f"✅ Found orderbook: trades={orderbook.get('tradesQuantity')}, volume={orderbook.get('collateralVolume')}")
    else:
        print("❌ No orderbook found")
    
    if results["marketData_query"].get("data", {}).get("marketData"):
        has_market_data = True
        market_data = results["marketData_query"]["data"]["marketData"]
        print(f"✅ Found market data: price={market_data.get('priceOrderbook')}")
    else:
        print("❌ No market data found")
    
    if not has_trades and not has_orderbook and not has_market_data:
        print("\nSUGGESTIONS:")
        print("- The token ID may be invalid or formatted incorrectly")
        print("- The token might not exist in this particular subgraph")
        print("- There might be a permission issue or API restriction")
        print("- Try running with --all-tests to get more detailed diagnostics")

if __name__ == "__main__":
    main()