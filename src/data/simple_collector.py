from subgrounds import Subgrounds
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv('GRAPH_API_KEY')

print("API Key:", api_key[:5] + "..." if api_key else "Not found")

# Initialize Subgrounds
sg = Subgrounds()

# Connect to orderbook subgraph
orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"

print("\nConnecting to orderbook subgraph...")
orderbook_subgraph = sg.load_subgraph(orderbook_url)

print("\nAttempting GraphQL query:")
print("""
query {
  orderbooks(
    first: 5
    orderBy: tradesQuantity
    orderDirection: desc
  ) {
    id
    tradesQuantity
    collateralVolume
  }
}
""")

# Query using Subgrounds - get top 5 orderbooks by trade volume
query = orderbook_subgraph.Query.orderbooks(
    first=5,
    orderBy='tradesQuantity',
    orderDirection='desc'
)

print("\nAttempting to query orderbook data...")
df = sg.query_df([
    query.id,
    query.tradesQuantity,
    query.collateralVolume
])

print("\nOrderbook Results:")
print(df)

if df.empty:
    print("No data found")
else:
    print("\nSuccessfully retrieved orderbook data")
    print(f"Number of orderbooks found: {len(df)}")
    
    # If we got data, print the first valid market ID we can use
    if 'orderbooks_id' in df.columns and not df['orderbooks_id'].isnull().all():
        valid_id = df['orderbooks_id'].iloc[0]
        print(f"\nFound valid market ID: {valid_id}")