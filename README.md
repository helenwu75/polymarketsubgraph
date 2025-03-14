Machine learning project analyzing Polymarket prediction market data using The Graph API.

Directory Structure:

- polymarket_raw_data/ - Main directory for raw collected data
- polymarket_raw_data/trades/ - Trade data for each token as Parquet files
- polymarket_raw_data/collection_summary.json - Collection statistics and results
- polymarket_raw_data/market_tokens.json - Mapping of markets to their token IDs

Useful Scripts:

- market_data_collector.py - Extracts raw trade data for Polymarket tokens from a csv (using clobTokenIds attribute)
- price_data.py - Calculates key pricing metrics for given tokens using stream processing (closing price, VWAP, price volatility)
- extract_metrics.py - Extracts comprehensive metrics for given market (average daily volume, length, # traders, trader-to-volume ratio, trading frequency, buy/sell)
