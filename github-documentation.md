# Polymarket Election Markets Analysis

This repository contains code and documentation for analyzing Polymarket election prediction markets using data from The Graph API. The analysis focuses on identifying factors that contribute to prediction accuracy and understanding market dynamics.

## Project Overview

This research analyzes 500 election markets from Polymarket to evaluate their predictive accuracy and identify key factors that correlate with successful predictions. Using data collected from Polymarket's subgraphs, we calculate comprehensive metrics related to price behavior, trading patterns, and trader characteristics.

### Key Findings

- Election prediction markets demonstrate remarkable accuracy (94.5% correct predictions)
- Presidential, Prime Minister, and General elections showed 100% prediction accuracy
- Parliamentary elections showed 95.5% accuracy
- Geographic distribution showed high accuracy across diverse political systems
- Late-stage trading activity negatively correlates with prediction accuracy
- Price volatility positively correlates with prediction accuracy

## Data Collection

Data is collected using subgraphs from The Graph API to access Polymarket data:

- **Market data**: Basic information about each market
- **Trade data**: Individual trade records for each market token
- **Orderbook data**: Aggregate trading statistics
- **Order events**: Detailed record of order matches

The data collection process is implemented in `market_data_collector.py` which extracts data for tokens associated with each market and stores it in the following structure:

```
polymarket_raw_data/
├── trades/                  # Trade data in Parquet format
├── orderbooks/              # Orderbook data in JSON format 
├── matched_events/          # Order event data in Parquet format
├── collection_summary.json  # Collection statistics
└── market_tokens.json       # Mapping of markets to tokens
```

## Metrics Calculation

We calculate a comprehensive set of metrics for each market using `election_market_metrics.py`. These metrics fall into four main categories:

### Price-Based Metrics

| Metric | Description | Formula/Method |
|--------|-------------|----------------|
| Closing Price | Last trade price before 24h prior to election | Last trade before cutoff |
| Price 2 Days Prior | Last trade price before 48h prior to election | Last trade before 48h cutoff |
| Pre-election VWAP (48h) | Volume-weighted average price in final 48 hours | `∑(price × size) / ∑(size)` |
| Price Volatility | Coefficient of variation of prices in final week | `std(price) / mean(price)` |
| Price Range | High-low range in final week | `max(price) - min(price)` |
| Final Week Momentum | Price change over final week | `last_price - first_price` |
| Price Fluctuations | Times price crossed 0.5 threshold in final week | Count of 0.5 crossings |
| Last Trade Price | Final price before market end | Last recorded trade price |

### Trading Activity Metrics

| Metric | Description | Formula/Method |
|--------|-------------|----------------|
| Market Duration Days | Active period of the market | End date - Start date |
| Trading Frequency | Average trades per day | `total_trades / market_duration` |
| Buy/Sell Ratio | Ratio of buys to sells | `buys_quantity / sells_quantity` |
| Trading Continuity | Percentage of days with trades | `trading_days / market_duration` |
| Late Stage Participation | Proportion of trades in final week | `final_week_trades / total_trades` |
| Volume Acceleration | Final week activity vs. overall | `(final_week_trades/7) / (total_trades/duration)` |

### Trader-Based Metrics

| Metric | Description | Formula/Method |
|--------|-------------|----------------|
| Unique Traders Count | Number of distinct traders | `len(unique_addresses)` |
| Trader to Trade Ratio | Trades per trader | `total_trades / unique_traders` |
| Two-Way Traders Ratio | Proportion who both bought and sold | `len(buyers ∩ sellers) / len(traders)` |
| Trader Concentration | Activity of top traders | `trades_by_top_10%_traders / total_trades` |
| New Trader Influx | Proportion of traders new in final week | `len(new_traders) / len(all_traders)` |

### Prediction Accuracy Metrics

| Metric | Description | Formula/Method |
|--------|-------------|----------------|
| Prediction Correct | Binary correctness indicator | For "Yes" outcome: `closing_price > 0.5`<br>For "No" outcome: `closing_price < 0.5` |
| Prediction Error | Absolute error in prediction | For "Yes" outcome: `\|1 - closing_price\|`<br>For "No" outcome: `\|0 - closing_price\|` |
| Prediction Confidence | Distance from 0.5 (scaled to [0,1]) | `\|closing_price - 0.5\| × 2` |

## Factor Importance Analysis

The analysis examines correlations between market metrics and prediction accuracy. Key correlations with prediction correctness include:

| Metric | Correlation | Interpretation |
|--------|-------------|----------------|
| Prediction Error | -0.8319 | Strong negative correlation (expected) |
| Late Stage Participation | -0.2323 | Moderate negative - late trading may reduce accuracy |
| Closing Price | -0.1717 | Modest negative - uncertain predictions less accurate |
| Price Volatility | 0.1326 | Modest positive - active price discovery may help |
| Two-way Traders Ratio | 0.1140 | Slight positive - traders on both sides improve accuracy |

## Running the Analysis

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, requests, tqdm, python-dotenv

### Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Graph API key: `GRAPH_API_KEY=your_api_key`
4. Create directories for data storage: `mkdir -p polymarket_raw_data/trades polymarket_raw_data/orderbooks polymarket_raw_data/matched_events`

### Data Collection

```bash
python market_data_collector.py --input top_election_markets.csv
```

### Metrics Calculation

```bash
python election_market_metrics.py --input top_election_markets.csv --output election_metrics_results.csv
```

### Analysis Options

- Use `--max-workers` to control parallelism
- Use `--sequential` for sequential processing
- Use `--max-markets` to limit the number of markets processed

## Future Work

- Factor analysis to identify the most important predictors of market accuracy
- Categorization by election type and country for more granular analysis
- Investigation of comment activity and its relationship to market accuracy
- Analysis of non-election markets for comparison

## License

[MIT License](LICENSE)

## Acknowledgments

This research uses data from Polymarket and The Graph API.
