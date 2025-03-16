# Polymarket Data Collector Documentation

## Overview

The Polymarket Data Collector is a Python-based tool designed to efficiently extract and store market data from Polymarket's subgraphs. It collects three main types of data:

1. **Trade Data** (enrichedOrderFilled events) - Detailed records of individual trades
2. **Orderbook Data** - Market-level statistics and volume information
3. **Order Events** - Both OrderFilledEvent and OrdersMatchedEvent data

This documentation provides information about the tool's usage, file organization, and data schemas to help future users understand and extend its functionality.

## Usage

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

The collector organizes data in the following structure:

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

## Data Schemas

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

Example row:

```
id: 0x9b57a5ac81adbb8ab47eb9b8e029667c9f7ee8862351fe0775f4ca55ddd47701_0x60e6c947d7721c7a4512d83011145a6c39ce914c569a228f2805f5665520f342
price: 0.998
side: Buy
size: 44910000
timestamp: 1734824599
transactionHash: 0x9b57a5ac81adbb8ab47eb9b8e029667c9f7ee8862351fe0775f4ca55ddd47701
maker_id: 0x55830fd5f60f5c42543c879868599cf0e24f72d7
taker_id: 0xc5d563a36ae78145c45a50134d48a1215220f80a
token_id: 14875660322078163696008576577608406432294409984642605754965521546683761493099
```

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

Example row:

```
fee: 0
id: 0x5696013c54d1685616421d68f0aef5aca1cdca59e63aa215aca669d18692e335_0x32b254d4a8ffb108860a91d6da03ce5c48f3360298e031d6fa2677dc28ce1197
makerAmountFilled: 301480000
makerAssetId: 24139171074793153246426895335330889394076802326509142035053601283282850503144
orderHash: 0x32b254d4a8ffb108860a91d6da03ce5c48f3360298e031d6fa2677dc28ce1197
takerAmountFilled: 299671120
takerAssetId: 0
timestamp: 1709254392
transactionHash: 0x5696013c54d1685616421d68f0aef5aca1cdca59e63aa215aca669d18692e335
source: OrderFilledEvent
event_type: OrderFilledEvent
token_id: 24139171074793153246426895335330889394076802326509142035053601283282850503144
```

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

Example JSON:

```json
{
  "buysQuantity": "1380",
  "collateralBuyVolume": "1017426364129",
  "collateralSellVolume": "301223321512",
  "collateralVolume": "1318649685641",
  "id": "80113082277751206069651919258603949325416407205034656804551579692845543370513",
  "lastActiveDay": "19735",
  "scaledCollateralBuyVolume": "1017426.364129",
  "scaledCollateralSellVolume": "301223.321512",
  "scaledCollateralVolume": "1318649.685641",
  "sellsQuantity": "509",
  "tradesQuantity": "1889"
}
```

### Summary Files (JSON Format)

The collector also generates summary files with collection statistics:

- **collection_summary.json**: Latest collection run summary
- **collection*summary*<timestamp>.json**: Timestamped collection run summaries
- **market_tokens.json**: Mapping of market IDs to their token IDs
- **market_id_to_question.json**: Mapping of market IDs to their question texts

These files provide metadata about the collection process and relationships between markets and tokens.

## GraphQL Subgraph Structure

The collector interfaces with Polymarket's subgraphs which have the following entity types:

### OrderFilledEvent Entity

```graphql
type OrderFilledEvent @entity {
  id: ID!
  transactionHash: Bytes!
  timestamp: BigInt!
  orderHash: Bytes!
  maker: String!
  taker: String!
  makerAssetId: String!
  takerAssetId: String!
  makerAmountFilled: BigInt!
  takerAmountFilled: BigInt!
  fee: BigInt!
}
```

### OrdersMatchedEvent Entity

```graphql
type OrdersMatchedEvent @entity {
  id: ID!
  timestamp: BigInt!
  makerAssetID: BigInt!
  takerAssetID: BigInt!
  makerAmountFilled: BigInt!
  takerAmountFilled: BigInt!
}
```

### Orderbook Entity

```graphql
type Orderbook @entity {
  id: ID!
  tradesQuantity: BigInt!
  buysQuantity: BigInt!
  sellsQuantity: BigInt!
  collateralVolume: BigInt!
  scaledCollateralVolume: BigDecimal!
  collateralBuyVolume: BigInt!
  scaledCollateralBuyVolume: BigDecimal!
  collateralSellVolume: BigInt!
  scaledCollateralSellVolume: BigDecimal!
  lastActiveDay: BigInt!
}
```

## Technical Details

### Dependencies

- Python 3.7+
- pandas
- numpy
- requests
- tqdm
- python-dotenv

### Environment Variables

- `GRAPH_API_KEY`: API key for accessing The Graph API (required)

### Performance Considerations

- The tool supports concurrent processing using Python's ThreadPoolExecutor
- Batch sizes are limited to 1000 records per query to comply with The Graph API limits
- Pagination is implemented to handle large datasets
- Client-side filtering is used for some queries to improve reliability

## Extending the Collector

To extend the collector for additional data types:

1. Create a new collection function following the pattern of existing ones
2. Add appropriate command-line arguments
3. Update the `process_token` function to include the new data type
4. Include the new data type in the summary statistics

## Troubleshooting

Common issues and solutions:

- **API Key Errors**: Ensure your `.env` file contains a valid `GRAPH_API_KEY`
- **Failed to Parse Query**: Verify token IDs are in the correct format for the query
- **Query Rate Limits**: Adjust batch size and max workers to prevent rate limiting
- **Missing Data**: Check the specific subgraph schema for field naming conventions

## Conclusion

The Polymarket Data Collector provides a flexible and efficient way to extract market data for analysis. By understanding its file structure and data schemas, researchers can effectively integrate this data into machine learning pipelines and analytical workflows.
