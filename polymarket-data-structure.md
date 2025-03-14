# Polymarket Data Structure Guide

This document outlines the data structure used in our machine learning project analyzing Polymarket election markets. It explains how data is collected, organized, and can be referenced for future queries.

## Directory Structure

The data is organized in the following directory structure:

- **polymarket_raw_data/** - Main directory for all collected data
- **polymarket_raw_data/trades/** - Contains trade data for each token as Parquet files
- **polymarket_raw_data/collection_summary.json** - Collection statistics and results
- **polymarket_raw_data/market_tokens.json** - Mapping of markets to their token IDs

## Data Collection Process

Our `market_data_collector.py` script executes the following workflow:

1. **Token Extraction**:

   - Extracts unique token IDs from an input CSV file containing top election markets
   - Uses the "clobTokenIds" column to identify relevant tokens

2. **Market-to-Token Mapping**:

   - Creates a JSON file mapping market questions to their associated token IDs
   - Stored in `market_tokens.json` for easy reference

3. **Trade Data Collection**:
   - For each token, queries the Polymarket orderbook subgraph using GraphQL
   - Collects all trades with pagination (in batches of 1000)
   - Processes each batch of trades efficiently
   - Stores the data in Parquet format for optimal read/write performance

## Data Structure Within Parquet Files

Each token's trade data is stored in a separate Parquet file containing the complete historical record of all trades for that token that were indexed by the Polymarket subgraph at the time of collection.

### Schema:

| Field             | Description                            |
| ----------------- | -------------------------------------- |
| `id`              | Unique identifier for the trade        |
| `timestamp`       | When the trade occurred                |
| `price`           | Price at which the trade executed      |
| `side`            | Buy or Sell                            |
| `size`            | Size of the trade                      |
| `maker_id`        | Address of the maker                   |
| `taker_id`        | Address of the taker                   |
| `transactionHash` | Blockchain transaction hash            |
| `token_id`        | The token ID (added during processing) |

### Data Completeness:

- Each Parquet file contains all historical trades for the token via exhaustive pagination
- The collection script retrieves all available trades with no time limitations
- Data is complete up to the collection timestamp; new trades require re-running the collection process

## Summary Data

The `collection_summary.json` file contains aggregated statistics:

- Timestamp of collection
- Number of tokens processed
- Number of successful token collections
- Number of tokens with trades
- Total trades collected
- Detailed results for each token processed

## Subgraph Data Relationships

Our data is sourced from multiple Polymarket subgraphs:

1. **Orderbook Subgraph** - Main source of trading data:

   - Trade history
   - Prices
   - Volume
   - Order book depth

2. **Activity Subgraph** - Contains:

   - Market creation data
   - Market creation timestamps
   - Conditions and positions

3. **PNL Subgraph** - Contains:
   - Profit and loss data
   - User positions

Each Polymarket market has:

- A unique market ID (FixedProductMarketMaker address)
- Associated token IDs (clobTokenIds) for each outcome
- Related conditions and positions linking everything together

## Accessing Data for Future Queries

For future queries, here's how to reference the structured data:

### Market-Level Analysis

```python
# Access market-level data
import json

# Load market-to-token mapping
with open('polymarket_raw_data/market_tokens.json', 'r') as f:
    market_tokens = json.load(f)

# Get tokens for a specific market
market_question = "Will Donald Trump win the 2024 US Presidential Election?"
market_token_ids = market_tokens.get(market_question, [])
```

### Token-Level Analysis

```python
# Access token-level trade data
import pandas as pd

# Load trade data for a specific token
token_id = "48331043336612883890938759509493159234755048973500640148014422747788308965732"
trades_df = pd.read_parquet(f"polymarket_raw_data/trades/{token_id}.parquet")

# Calculate basic metrics
vwap = (trades_df['price'] * trades_df['size']).sum() / trades_df['size'].sum()
total_volume = trades_df['size'].sum()
trade_count = len(trades_df)
```

### Combined Analysis

By leveraging the market-to-token mapping, you can perform comprehensive analysis across all tokens in a market:

```python
import pandas as pd
import json

# Load market-to-token mapping
with open('polymarket_raw_data/market_tokens.json', 'r') as f:
    market_tokens = json.load(f)

# Select a market
market_question = "Will Donald Trump win the 2024 US Presidential Election?"
token_ids = market_tokens.get(market_question, [])

# Analyze all tokens in the market
market_data = []
for token_id in token_ids:
    try:
        token_df = pd.read_parquet(f"polymarket_raw_data/trades/{token_id}.parquet")
        # Add token_id as reference
        token_df['token_id'] = token_id
        market_data.append(token_df)
    except Exception as e:
        print(f"Error loading {token_id}: {e}")

# Combine all token data for the market
if market_data:
    combined_df = pd.concat(market_data)
    # Now you can analyze the entire market
    print(f"Total trades: {len(combined_df)}")
    print(f"Total volume: {combined_df['size'].sum()}")
    print(f"Unique traders: {len(combined_df['maker_id'].unique()) + len(combined_df['taker_id'].unique())}")
```

## Additional Tools

Our repository includes several scripts for analyzing Polymarket data:

- **market_data_collector.py** - Extracts raw trade data for Polymarket tokens
- **price_data.py** - Calculates key pricing metrics (closing price, VWAP, price volatility)
- **extract_metrics.py** - Provides comprehensive market metrics (daily volume, length, trader statistics)

These tools can be combined to create a powerful pipeline for Polymarket data analysis, forming the foundation for machine learning model development.
