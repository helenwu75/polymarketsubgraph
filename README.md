# Polymarket Subgraph Data Analysis

This repository contains tools for collecting, processing, and analyzing Polymarket prediction market data using The Graph API. It focuses on extracting comprehensive market data and calculating metrics for prediction accuracy analysis.

Creator: Helen Wu

## Project Overview

This research analyzes election markets from Polymarket to evaluate their predictive accuracy and identify key factors that correlate with successful predictions. Using data collected from Polymarket's subgraphs, we calculate comprehensive metrics related to price behavior, trading patterns, and trader characteristics.

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your Graph API key:

```
GRAPH_API_KEY=your_api_key_here
```

## Data Collection

The `market_data_collector.py` script efficiently extracts comprehensive market data from Polymarket subgraphs including:

1. **Trade data** (enrichedOrderFilled events) - Detailed records of individual trades
2. **Orderbook data** - Market-level statistics and volume information
3. **Order events** - Both OrderFilledEvent and OrdersMatchedEvent data

### Usage

```bash
python market_data_collector.py --input <csv_file> [options]
```

### Command Line Arguments

| Argument                | Description                          | Default             |
| ----------------------- | ------------------------------------ | ------------------- |
| `--input`               | CSV file with market data (required) | -                   |
| `--column`              | Column name with token IDs           | clobTokenIds        |
| `--max-tokens`          | Maximum number of tokens to process  | All                 |
| `--max-workers`         | Maximum parallel workers             | 8                   |
| `--sequential`          | Process tokens sequentially          | False               |
| `--skip-trades`         | Skip trade data collection           | False               |
| `--skip-orderbooks`     | Skip orderbook data collection       | False               |
| `--skip-matched-events` | Skip matched events collection       | False               |
| `--timeout`             | Request timeout in seconds           | 60                  |
| `--batch-size`          | Batch size for queries               | 1000                |
| `--output-dir`          | Base output directory                | polymarket_raw_data |

### Example Commands

Basic usage:

```bash
python market_data_collector.py --input top_election_markets.csv
```

Collect only order events:

```bash
python market_data_collector.py --input top_election_markets.csv --skip-trades --skip-orderbooks
```

Process a limited number of tokens:

```bash
python market_data_collector.py --input top_election_markets.csv --max-tokens 10
```

## Directory Structure

The data collection process creates the following directory structure:

```
polymarket_raw_data/
├── trades/                      # Trade data in Parquet format
│   ├── <token_id>.parquet       # One file per token
│   └── ...
├── orderbooks/                  # Orderbook data in JSON format
│   ├── <token_id>.json          # One file per token
│   └── ...
├── matched_events/              # Order event data in Parquet format
│   ├── <token_id>.parquet       # One file per token
│   └── ...
├── collection_summary.json      # Latest collection summary
├── collection_summary_<timestamp>.json  # Timestamped collection summaries
├── market_tokens.json           # Mapping of markets to token IDs
└── market_id_to_question.json   # Mapping of market IDs to questions
```

## Metrics Calculation

After collecting the raw data, you can calculate comprehensive metrics for each market using `election_market_metrics.py`. These metrics fall into four main categories:

### Usage

```bash
python election_market_metrics.py --input markets.csv --output metrics.csv
```

### Command Line Arguments

| Argument        | Description                                | Default                     |
| --------------- | ------------------------------------------ | --------------------------- |
| `--input`       | Input CSV file with market data (required) | -                           |
| `--output`      | Output CSV filename                        | election_market_metrics.csv |
| `--max-workers` | Maximum parallel workers                   | 8                           |
| `--sequential`  | Process markets sequentially               | False                       |
| `--max-markets` | Maximum number of markets to process       | All                         |

### Price-Based Metrics

| Metric                  | Description                                      | Formula/Method               |
| ----------------------- | ------------------------------------------------ | ---------------------------- |
| Closing Price           | Last trade price before 24h prior to election    | Last trade before cutoff     |
| Price 2 Days Prior      | Last trade price before 48h prior to election    | Last trade before 48h cutoff |
| Pre-election VWAP (48h) | Volume-weighted average price in final 48 hours  | `∑(price × size) / ∑(size)`  |
| Price Volatility        | Coefficient of variation of prices in final week | `std(price) / mean(price)`   |
| Price Range             | High-low range in final week                     | `max(price) - min(price)`    |
| Final Week Momentum     | Price change over final week                     | `last_price - first_price`   |
| Price Fluctuations      | Times price crossed 0.5 threshold in final week  | Count of 0.5 crossings       |
| Last Trade Price        | Final price before market end                    | Last recorded trade price    |

### Trading Activity Metrics

| Metric                   | Description                        | Formula/Method                                    |
| ------------------------ | ---------------------------------- | ------------------------------------------------- |
| Market Duration Days     | Active period of the market        | End date - Start date                             |
| Trading Frequency        | Average trades per day             | `total_trades / market_duration`                  |
| Buy/Sell Ratio           | Ratio of buys to sells             | `buys_quantity / sells_quantity`                  |
| Trading Continuity       | Percentage of days with trades     | `trading_days / market_duration`                  |
| Late Stage Participation | Proportion of trades in final week | `final_week_trades / total_trades`                |
| Volume Acceleration      | Final week activity vs. overall    | `(final_week_trades/7) / (total_trades/duration)` |

### Trader-Based Metrics

| Metric                | Description                             | Formula/Method                             |
| --------------------- | --------------------------------------- | ------------------------------------------ |
| Unique Traders Count  | Number of distinct traders              | `len(unique_addresses)`                    |
| Trader to Trade Ratio | Trades per trader                       | `total_trades / unique_traders`            |
| Two-Way Traders Ratio | Proportion who both bought and sold     | `len(buyers ∩ sellers) / len(traders)`     |
| Trader Concentration  | Activity of top traders                 | `trades_by_top_10%_traders / total_trades` |
| New Trader Influx     | Proportion of traders new in final week | `len(new_traders) / len(all_traders)`      |

### Prediction Accuracy Metrics

| Metric                | Description                         | Formula/Method                                                                          |
| --------------------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| Prediction Correct    | Binary correctness indicator        | For "Yes" outcome: `closing_price > 0.5`<br>For "No" outcome: `closing_price < 0.5`     |
| Prediction Error      | Absolute error in prediction        | For "Yes" outcome: `\|1 - closing_price\|`<br>For "No" outcome: `\|0 - closing_price\|` |
| Prediction Confidence | Distance from 0.5 (scaled to [0,1]) | `\|closing_price - 0.5\| × 2`                                                           |

## Data Structure Details

### Trade Data (Parquet Format)

Trade data is stored in Parquet files under the `trades/` directory with the following schema:

| Column          | Type   | Description                                             |
| --------------- | ------ | ------------------------------------------------------- |
| id              | string | Unique trade identifier (transaction hash + order hash) |
| price           | string | Trade execution price                                   |
| side            | string | Trade side (Buy/Sell)                                   |
| size            | string | Trade size in base units                                |
| timestamp       | string | Unix timestamp of the trade                             |
| transactionHash | string | Blockchain transaction hash                             |
| maker_id        | string | Address of the maker                                    |
| taker_id        | string | Address of the taker                                    |
| token_id        | string | Token ID reference                                      |

### Order Events Data (Parquet Format)

Order events data is stored in Parquet files under the `matched_events/` directory with the following schema:

| Column            | Type   | Description                    |
| ----------------- | ------ | ------------------------------ |
| fee               | int64  | Fee paid by the order maker    |
| id                | string | Unique event identifier        |
| makerAmountFilled | int64  | Maker amount filled            |
| makerAssetId      | string | Maker asset ID                 |
| orderHash         | string | Order hash                     |
| takerAmountFilled | int64  | Taker amount filled            |
| takerAssetId      | string | Taker asset ID                 |
| timestamp         | int64  | Unix timestamp of the event    |
| transactionHash   | string | Blockchain transaction hash    |
| source            | string | Data source (OrderFilledEvent) |
| event_type        | string | Event type (OrderFilledEvent)  |
| token_id          | string | Token ID reference             |

### Orderbook Data (JSON Format)

Orderbook data is stored in JSON files under the `orderbooks/` directory with the following schema:

| Field                      | Type   | Description                        |
| -------------------------- | ------ | ---------------------------------- |
| id                         | string | Token ID                           |
| tradesQuantity             | string | Total number of trades             |
| buysQuantity               | string | Number of buy trades               |
| sellsQuantity              | string | Number of sell trades              |
| collateralVolume           | string | Total trading volume in base units |
| scaledCollateralVolume     | string | Total trading volume scaled (USDC) |
| collateralBuyVolume        | string | Buy volume in base units           |
| scaledCollateralBuyVolume  | string | Buy volume scaled (USDC)           |
| collateralSellVolume       | string | Sell volume in base units          |
| scaledCollateralSellVolume | string | Sell volume scaled (USDC)          |
| lastActiveDay              | string | Last active day                    |

## Subgraph Structure

This project interfaces with three main Polymarket subgraphs:

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

## Accessing Data for Analysis

You can access the collected data for your analysis as follows:

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

## Statistics and Visualization

For visualization and summary statistics generation, you can use `summary_stats.py`:

```bash
python summary_stats.py --input election_metrics_results.csv
```

This will generate various visualizations and report files in the `summary_stats_results` directory.

## Factor Importance Analysis

The analysis examines correlations between market metrics and prediction accuracy. Key correlations with prediction correctness include:

| Metric                   | Correlation | Interpretation                                           |
| ------------------------ | ----------- | -------------------------------------------------------- |
| Prediction Error         | -0.8319     | Strong negative correlation (expected)                   |
| Late Stage Participation | -0.2323     | Moderate negative - late trading may reduce accuracy     |
| Closing Price            | -0.1717     | Modest negative - uncertain predictions less accurate    |
| Price Volatility         | 0.1326      | Modest positive - active price discovery may help        |
| Two-way Traders Ratio    | 0.1140      | Slight positive - traders on both sides improve accuracy |

## Project Structure

```
.
├── data_cleaning.py               # Data preprocessing for ML
├── election_market_metrics.py     # Metrics calculation
├── market_data_collector.py       # Raw data collection
├── requirements.txt               # Project dependencies
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── config/                    # Configuration utilities
│   ├── data/                      # Data collection modules
│   ├── models/                    # Analysis models
│   └── utils/                     # Utility functions
├── append_brier_scores.py         # Add prediction scoring
└── summary_stats.py               # Generate reports and visualizations
```

## Acknowledgments

This research uses data from Polymarket and The Graph API.
