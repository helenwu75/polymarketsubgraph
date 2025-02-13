# Polymarket Data Relationships and Patterns

## Entity Relationships

### Market Structure
```
FixedProductMarketMaker
├── Condition (1:1)
├── Collateral (1:1)
└── Positions (1:Many)
    └── UserPositions (Many:Many)
```

### Trading Activity
```
Account
├── MarketPositions (1:Many)
├── Transactions (1:Many)
└── MarketProfits (1:Many)

OrderFilledEvent
├── Maker (Account)
├── Taker (Account)
└── Market (MarketData)
```

### PNL Tracking
```
UserPosition
├── User (Account)
└── Token (Position)
    └── Condition (1:1)
```

## Data Flow Patterns

### Market Lifecycle
1. Market Creation
   - FixedProductMarketMaker created
   - Condition linked
   - Initial liquidity added

2. Trading Activity
   - OrderFilledEvents generated
   - UserPositions updated
   - Market volumes updated

3. Market Resolution
   - Condition resolved
   - Payouts determined
   - Redemptions processed

### Position Updates
1. Position Opening
   - Split or direct purchase
   - UserPosition created/updated
   - Account metrics updated

2. Position Modification
   - Additional trades
   - Average price recalculated
   - Volume metrics updated

3. Position Closing
   - Merge or sale
   - RealizedPnL calculated
   - Final metrics updated

## Common Query Patterns

### Market Analysis
1. Volume Analysis
   - Daily/hourly volume
   - Buy/sell ratio
   - Price trends

2. Liquidity Analysis
   - Market maker reserves
   - Order book depth
   - Trading activity

3. User Behavior
   - Position sizes
   - Trading frequency
   - Profit/loss patterns

## Data Processing Considerations

### Time-based Analysis
1. Block Time vs Timestamp
   - Block numbers for historical queries
   - Timestamps for time-based analysis
   - Consider network congestion

2. Data Aggregation
   - Daily summaries
   - Rolling averages
   - Volume-weighted metrics

3. Data Synchronization
   - Cross-subgraph data alignment
   - Consistent time boundaries
   - Entity relationship mapping

### Performance Optimization

1. Query Efficiency
   - Minimize nested queries
   - Use appropriate indexes
   - Batch related queries

2. Data Caching
   - Cache static data
   - Update frequency
   - Cache invalidation

3. Rate Limiting
   - Request batching
   - Parallel queries
   - Backoff strategies

## Common Metrics Calculation

### Market Metrics
```python
def calculate_market_metrics(market_data):
    return {
        'volume': market_data['collateralVolume'],
        'trades': market_data['tradesQuantity'],
        'liquidity': calculate_liquidity(market_data),
        'price_impact': calculate_price_impact(market_data)
    }
```

### Position Metrics
```python
def calculate_position_metrics(position_data):
    return {
        'pnl': position_data['realizedPnl'],
        'avg_price': position_data['avgPrice'],
        'current_value': calculate_current_value(position_data),
        'roi': calculate_roi(position_data)
    }
```

### Trading Metrics
```python
def calculate_trading_metrics(trading_data):
    return {
        'volume': trading_data['collateralVolume'],
        'frequency': calculate_trade_frequency(trading_data),
        'success_rate': calculate_success_rate(trading_data),
        'avg_position_size': calculate_avg_position(trading_data)
    }
```
