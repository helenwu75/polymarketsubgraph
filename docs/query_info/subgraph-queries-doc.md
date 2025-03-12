# Polymarket Subgraph Query Documentation

## Table of Contents
1. [General Query Guidelines](#general-query-guidelines)
2. [Market Activity Subgraph](#market-activity-subgraph)
3. [Orderbook Subgraph](#orderbook-subgraph)
4. [PNL (Profit and Loss) Subgraph](#pnl-subgraph)
5. [Best Practices](#best-practices)

## General Query Guidelines

### Authentication
```python
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}
```

### Pagination
- Use `first` parameter to limit results (max 1000)
- Use `skip` or `cursor` for pagination
- Always order results for consistent pagination

### Common Parameters
- `orderBy`: Field to sort by
- `orderDirection`: 'asc' or 'desc'
- `where`: Filter conditions
- `block`: Historical data at specific block

## Market Activity Subgraph

### Key Entities

#### 1. FixedProductMarketMaker
```graphql
{
  fixedProductMarketMakers(
    first: 100
    orderBy: creationTimestamp
    orderDirection: desc
  ) {
    id                    # Market maker address
    creator              # Address which deployed this market
    creationTimestamp    # Deployment time
    creationTransactionHash
    conditionalTokenAddress
    conditions {
      id
    }
    fee                  # Trading fee percentage
    tradesQuantity       # Total number of trades
    collateralVolume     # Total trading volume
  }
}
```

#### 2. Transaction Data
```graphql
{
  splits(
    first: 100
    orderBy: timestamp
    where: { timestamp_gt: "1642000000" }
  ) {
    id
    timestamp
    stakeholder
    condition
    amount
  }
}
```

#### 3. Market Conditions
```graphql
{
  conditions(
    first: 100
    where: { id: "0x..." }
  ) {
    id
    fixedProductMarketMakers {
      id
      tradesQuantity
    }
  }
}
```

## Orderbook Subgraph

### Key Entities

#### 1. Account Activity
```graphql
{
  accounts(
    first: 100
    orderBy: lastTradedTimestamp
    orderDirection: desc
  ) {
    id                      # User address
    collateralVolume        # Total trading volume
    scaledCollateralVolume  # Volume scaled by decimals
    numTrades              # Total number of trades
    profit                 # Total profit/loss
    lastTradedTimestamp
    marketPositions {
      market {
        id
      }
      netQuantity
      netValue
    }
  }
}
```

#### 2. Market Data
```graphql
{
  marketDatas(
    first: 100
    orderBy: priceOrderbook
    orderDirection: desc
  ) {
    id
    condition {
      id
    }
    fpmm {
      id
      collateralVolume
    }
    priceOrderbook      # Most recent price
    outcomeIndex
  }
}
```

#### 3. Trade History
```graphql
{
  orderFilledEvents(
    first: 100
    orderBy: timestamp
    where: { timestamp_gt: "1642000000" }
  ) {
    id
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
```

#### 4. Global Statistics
```graphql
{
  globals(first: 1) {
    id
    tradesQuantity
    collateralVolume
    scaledCollateralVolume
    numTraders
    numConditions
    numOpenConditions
  }
}
```

## PNL Subgraph

### Key Entities

#### 1. User Positions
```graphql
{
  userPositions(
    first: 100
    orderBy: realizedPnl
    orderDirection: desc
  ) {
    id                # User Address + Token ID
    user              # User address
    tokenId           # Token ID
    amount           # Current holding amount
    avgPrice         # Average entry price
    realizedPnl      # Realized profit/loss
    totalBought      # Total amount bought
  }
}
```

#### 2. Condition Data
```graphql
{
  conditions(first: 100) {
    id
    positionIds      # Associated token IDs
    payoutNumerators # Resolution values
    payoutDenominator
  }
}
```

#### 3. FPMM (Fixed Product Market Maker)
```graphql
{
  fpmms(first: 100) {
    id              # FPMM address
    conditionId     # Associated condition
  }
}
```

## Best Practices

### 1. Query Optimization
- Only request needed fields
- Use appropriate pagination size
- Include relevant filters
- Order results consistently

### 2. Error Handling
```python
try:
    response = requests.post(
        subgraph_url,
        headers=headers,
        json={'query': query},
        timeout=10
    )
    if response.status_code == 200:
        data = response.json()
        if 'errors' in data:
            handle_graphql_errors(data['errors'])
        return data['data']
except Exception as e:
    handle_request_error(e)
```

### 3. Rate Limiting
```python
import time
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=30, period=60)  # 30 calls per minute
def query_subgraph(query: str) -> dict:
    # Implementation
    pass
```

### 4. Pagination Example
```python
def fetch_all_results(query_template: str) -> List[dict]:
    results = []
    skip = 0
    while True:
        query = query_template.format(skip=skip)
        data = query_subgraph(query)
        if not data or not data['data']:
            break
        results.extend(data['data'])
        if len(data['data']) < 1000:  # Less than max results
            break
        skip += 1000
    return results
```

### 5. Data Validation
```python
def validate_market_data(data: dict) -> bool:
    required_fields = [
        'id', 'creationTimestamp', 'collateralVolume'
    ]
    return all(field in data for field in required_fields)
```

### 6. Historical Data Queries
```graphql
{
  fixedProductMarketMakers(
    block: {
      number: 15000000
    }
  ) {
    id
    collateralVolume
    tradesQuantity
  }
}
```

### 7. Common Issues and Solutions

#### Missing Data
- Always check for null values
- Use fallback values where appropriate
- Log missing data patterns

#### Rate Limiting
- Implement exponential backoff
- Cache frequently accessed data
- Batch queries where possible

#### Consistency
- Use consistent timestamps
- Handle different decimal scales
- Normalize entity IDs

### 8. Query Templates

#### Market Overview
```python
MARKET_OVERVIEW_QUERY = """
{
  fixedProductMarketMakers(
    first: 1000
    skip: {skip}
    orderBy: creationTimestamp
    orderDirection: desc
  ) {
    id
    creationTimestamp
    collateralVolume
    tradesQuantity
    outcomeTokenPrices
  }
}
"""
```

#### Trading Activity
```python
TRADING_ACTIVITY_QUERY = """
{
  accounts(
    first: 1000
    skip: {skip}
    orderBy: lastTradedTimestamp
    orderDirection: desc
    where: {
      numTrades_gt: 0
    }
  ) {
    id
    collateralVolume
    numTrades
    profit
    lastTradedTimestamp
  }
}
"""
```

#### Price History
```python
PRICE_HISTORY_QUERY = """
{
  orderFilledEvents(
    first: 1000
    skip: {skip}
    orderBy: timestamp
    orderDirection: asc
    where: {
      timestamp_gt: {start_time}
      timestamp_lt: {end_time}
    }
  ) {
    timestamp
    price
    makerAmountFilled
    takerAmountFilled
  }
}
"""
```
