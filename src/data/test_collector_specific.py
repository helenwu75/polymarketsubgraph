from subgrounds import Subgrounds
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv('GRAPH_API_KEY')

print("API Key:", api_key[:5] + "..." if api_key else "Not found")

# Initialize Subgrounds
sg = Subgrounds()

# Connect to both subgraphs
activity_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
orderbook_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/81Dm16JjuFSrqz813HysXoUPvzTwE7fsfPk2RTf66nyC"

print("\nConnecting to subgraphs...")
activity_subgraph = sg.load_subgraph(activity_url)
orderbook_subgraph = sg.load_subgraph(orderbook_url)

# Use the market ID you provided
market_id = "0x002d8d7bcd9f9d2ba629a47982baa9230466c5f5"

# Print the GraphQL queries we're attempting
print("\nAttempting GraphQL queries:")
print("""
# Activity Subgraph Query
query {
  fixedProductMarketMaker(id: "%s") {
    id
  }
}

# Orderbook Subgraph Query
query {
  fixedProductMarketMaker(id: "%s") {
    collateralVolume
  }
}
""" % (market_id, market_id))

# Query from activity subgraph
activity_query = activity_subgraph.Query.fixedProductMarketMaker(
    id=market_id
)

# Query from orderbook subgraph - let's just try collateralVolume first
orderbook_query = orderbook_subgraph.Query.fixedProductMarketMaker(
    id=market_id
)

print("\nAttempting to query market data...")

# Get data from each subgraph
activity_df = sg.query_df([activity_query.id])
orderbook_df = sg.query_df([orderbook_query.collateralVolume])

print("\nActivity Subgraph Result:")
print(activity_df)

print("\nOrderbook Subgraph Result:")
print(orderbook_df)

# Add error checking
if activity_df.empty and orderbook_df.empty:
    print("No data found in either subgraph")
elif activity_df.empty:
    print("No data found in activity subgraph")
elif orderbook_df.empty:
    print("No data found in orderbook subgraph")
else:
    print("\nSuccessfully retrieved market data from both subgraphs")