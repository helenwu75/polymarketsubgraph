# Orderbook Subgraph Schema Documentation

Generated on: 2025-02-12 21:33:54

## Account

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| collateralVolume | `BigInt` | Yes | Total volume of this user's trades in USDC base units |
| creationTimestamp | `BigInt` | Yes | Timestamp at which account first interacted with Polymarket |
| fpmmPoolMemberships | `[None]` | Yes | Markets in which user has provided liquidity |
| id | `ID` | Yes | User address |
| lastSeenTimestamp | `BigInt` | Yes | Timestamp at which account most recently interacted with Polymarket |
| lastTradedTimestamp | `BigInt` | Yes | Timestamp of last Buy or Sell transaction |
| marketPositions | `[None]` | Yes | Markets in which the user has taken a position on the outcome |
| marketProfits | `[None]` | Yes | Profits in USDC base units by market |
| merges | `[None]` | Yes | Merge of more specific outcome tokens into collateral / more general outcome tokens |
| numTrades | `BigInt` | Yes | Total number of trades performed by this user |
| profit | `BigInt` | Yes | Profit generated from fpmm and orderbook trades, merges and redemptions |
| redemptions | `[None]` | Yes | Redemption of underlying collateral after a market has resolved |
| scaledCollateralVolume | `BigDecimal` | Yes | Total volume of this user's trades in USDC scaled by 10^6 |
| scaledProfit | `BigDecimal` | Yes | Realized profit in USDC scaled by 10^6 |
| splits | `[None]` | Yes | Split of collateral / outcome tokens into multiple positions |
| transactions | `[None]` | Yes | Purchases and sales of shares by the user |

## Collateral

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| decimals | `Int` | Yes |  |
| id | `ID` | Yes | Token address |
| name | `String` | Yes |  |
| symbol | `String` | Yes |  |

## Condition

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| fixedProductMarketMakers | `None` | Yes | Market makers which are trading on this condition |
| id | `ID` | Yes |  |
| oracle | `Bytes` | Yes | Address which can resolve this condition |
| outcomeSlotCount | `Int` | Yes | Number of possible outcomes for this condition |
| payoutDenominator | `BigInt` | No |  |
| payoutNumerators | `[None]` | Yes |  |
| payouts | `[None]` | Yes | Fraction of collateral assigned to each outcome |
| questionId | `Bytes` | Yes | Question ID which corresponds to this condition |
| resolutionHash | `Bytes` | No | Hash of the resolution transaction |
| resolutionTimestamp | `BigInt` | No | Timestamp at which this condition was resolved |

## EnrichedOrderFilled

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| id | `ID` | Yes | Transaction hash + Order hash |
| maker | `Account` | Yes | Addresses of the maker and the taker |
| market | `Orderbook` | Yes | Market/CTF Token ID which the transaction is interacting with |
| orderHash | `Bytes` | Yes | Order hash |
| price | `BigDecimal` | Yes | Price of the conditional token |
| side | `TradeType` | Yes | Buy or Sell transaction |
| size | `BigInt` | Yes | Amount of collateral in trade |
| taker | `Account` | Yes |  |
| timestamp | `BigInt` | Yes | Timestamp at which transaction occurred |
| transactionHash | `Bytes` | Yes | Transaction hash |

## FixedProductMarketMaker

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| buysQuantity | `BigInt` | Yes | Number of purchases of shares from this market maker |
| collateralBuyVolume | `BigInt` | Yes | Global volume of share purchases in USDC base units |
| collateralSellVolume | `BigInt` | Yes | Global volume of share sales in USDC base units |
| collateralToken | `Collateral` | Yes | Token which is colleralising this market |
| collateralVolume | `BigInt` | Yes | Market volume in terms of the underlying collateral value |
| conditionalTokenAddress | `String` | Yes | Conditional Token Address |
| conditions | `None` | Yes | Conditions which this market is trading against |
| creationTimestamp | `BigInt` | Yes | Time at which this market was deployed |
| creationTransactionHash | `Bytes` | Yes | Hash of deployment transactions |
| creator | `Bytes` | Yes | Address which deployed this market |
| fee | `BigInt` | Yes | Percentage fee of trades taken by market maker. A 2% fee is represented as 2*10^16 |
| feeVolume | `BigInt` | Yes | Fees collected in terms of the underlying collateral value |
| id | `ID` | Yes | Market maker address |
| lastActiveDay | `BigInt` | Yes | Timestamp of last day during which someone made a trade |
| liquidityAddQuantity | `BigInt` | Yes | Number of times liquidity has been added to this market maker |
| liquidityParameter | `BigInt` | Yes | Constant product parameter k |
| liquidityRemoveQuantity | `BigInt` | Yes | Number of times liquidity has been removed from this market maker |
| outcomeSlotCount | `Int` | No | Number of outcomes which this market maker is trading |
| outcomeTokenAmounts | `None` | Yes | Balances of each outcome token held by the market maker |
| outcomeTokenPrices | `None` | Yes | Prices at which market maker values each outcome token |
| poolMembers | `[None]` | Yes | Addresses which are supplying liquidity to the market maker |
| scaledCollateralBuyVolume | `BigDecimal` | Yes | Global volume of share purchases in USDC scaled by 10^6 |
| scaledCollateralSellVolume | `BigDecimal` | Yes | Global volume of share sales in USDC scaled by 10^6 |
| scaledCollateralVolume | `BigDecimal` | Yes | Volume scaled by the number of decimals of collateralToken |
| scaledFeeVolume | `BigDecimal` | Yes | Fees scaled by the number of decimals of collateralToken |
| scaledLiquidityParameter | `BigDecimal` | Yes |  |
| sellsQuantity | `BigInt` | Yes | Number of sales of shares to this market maker |
| totalSupply | `BigInt` | Yes | Number of shares for tokens in the market maker's reserves |
| tradesQuantity | `BigInt` | Yes | Number of trades of any kind against this market maker |

## FpmmFundingAddition

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amountsAdded | `None` | Yes | Outcome tokens amounts added to FPMM |
| amountsRefunded | `None` | Yes | Outcome tokens amounts refunded to funder |
| fpmm | `FixedProductMarketMaker` | Yes | FPMM to which funding is being added |
| funder | `Account` | Yes | Account adding funding |
| id | `ID` | Yes | Transaction Hash |
| sharesMinted | `BigInt` | Yes | Liquidity shares minted to funder |
| timestamp | `BigInt` | Yes | Timestamp at which funding addition occurred |

## FpmmFundingRemoval

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amountsRemoved | `None` | Yes | Outcome tokens amounts removed from FPMM |
| collateralRemoved | `BigInt` | Yes |  |
| fpmm | `FixedProductMarketMaker` | Yes | FPMM to which funding is being removed |
| funder | `Account` | Yes | Account removing funding |
| id | `ID` | Yes | Transaction Hash |
| sharesBurnt | `BigInt` | Yes | Liquidity shares burned by funder |
| timestamp | `BigInt` | Yes | Timestamp at which funding removal occurred |

## FpmmPoolMembership

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amount | `BigInt` | Yes | Amount of liquidity tokens owned by funder |
| funder | `Account` | Yes | Account which is providing funding |
| id | `ID` | Yes | funder address + pool address |
| pool | `FixedProductMarketMaker` | Yes | Market to which funder is providing funding |

## Global

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| buysQuantity | `BigInt` | Yes | Number of purchases of shares from any market maker |
| collateralBuyVolume | `BigInt` | Yes | Global volume of share purchases in USDC base units |
| collateralFees | `BigInt` | Yes | Global fees in USDC base units |
| collateralSellVolume | `BigInt` | Yes | Global volume of share sales in USDC base units |
| collateralVolume | `BigInt` | Yes | Global volume in USDC base units |
| id | `ID` | Yes | ID is empty string, this is a singleton |
| numClosedConditions | `Int` | Yes |  |
| numConditions | `Int` | Yes |  |
| numOpenConditions | `Int` | Yes |  |
| numTraders | `BigInt` | Yes | Number of unique traders interacting with Polymarket |
| scaledCollateralBuyVolume | `BigDecimal` | Yes | Global volume of share purchases in USDC scaled by 10^6 |
| scaledCollateralFees | `BigDecimal` | Yes | Global fees in USDC scaled by 10^6 |
| scaledCollateralSellVolume | `BigDecimal` | Yes | Global volume of share sales in USDC scaled by 10^6 |
| scaledCollateralVolume | `BigDecimal` | Yes | Global volume in USDC scaled by 10^6 |
| sellsQuantity | `BigInt` | Yes | Number of sales of shares to any market maker |
| tradesQuantity | `BigInt` | Yes | Number of trades of any kind for all market makers |

## MarketData

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| condition | `Condition` | Yes | Condition that the token is linked to |
| fpmm | `FixedProductMarketMaker` | No | The linked FixedProductMarketMaker |
| id | `ID` | Yes | ERC1155 TokenID of the CTF Asset |
| outcomeIndex | `BigInt` | No | Outcome Index, may not be present if an FPMM is not created |
| priceOrderbook | `BigDecimal` | No | The most recent onchain price of the asset on the orderbook |

## MarketPosition

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| feesPaid | `BigInt` | Yes | Total amount of fees paid by user in relation to this position |
| id | `ID` | Yes |  |
| market | `MarketData` | Yes | Market/tokenId on which this position is on |
| netQuantity | `BigInt` | Yes | Number of outcome shares that the user current has |
| netValue | `BigInt` | Yes | Total value paid by the user to enter this position |
| quantityBought | `BigInt` | Yes | Number of outcome shares that the user has ever bought |
| quantitySold | `BigInt` | Yes | Number of outcome shares that the user has ever sold |
| user | `Account` | Yes | Address which holds this position |
| valueBought | `BigInt` | Yes | Total value of outcome shares that the user has bought |
| valueSold | `BigInt` | Yes | Total value of outcome shares that the user has sold |

## MarketProfit

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| condition | `Condition` | Yes | The ConditionID, used as the link between YES/NO tokens |
| id | `ID` | Yes | Keyed on ConditionID + user |
| profit | `BigInt` | Yes | Profit in USDC base units per market per account |
| scaledProfit | `BigDecimal` | Yes | Profit in USDC scaled by 10^6 |
| user | `Account` | Yes | User address |

## Merge

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amount | `BigInt` | Yes | The amount of outcome tokens being merged |
| collateralToken | `Collateral` | Yes | Token which is collateralising positions being merged |
| condition | `Condition` | Yes | Condition on which merge is occuring |
| id | `ID` | Yes | Transaction Hash |
| parentCollectionId | `Bytes` | Yes |  |
| partition | `None` | Yes |  |
| stakeholder | `Account` | Yes | Address which is performing this merge |
| timestamp | `BigInt` | Yes | Timestamp at which merge occurred |

## OrderFilledEvent

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| fee | `BigInt` | Yes | Fee paid by the order maker |
| id | `ID` | Yes | Transaction hash + Order hash |
| maker | `Account` | Yes | Addresses of the maker and the taker |
| makerAmountFilled | `BigInt` | Yes | Maker amount filled |
| makerAssetId | `String` | Yes | Maker assetId |
| orderHash | `Bytes` | Yes |  |
| taker | `Account` | Yes |  |
| takerAmountFilled | `BigInt` | Yes | Taker amount filled |
| takerAssetId | `String` | Yes | Taker assetId |
| timestamp | `BigInt` | Yes | Timestamp at which filled occurred |
| transactionHash | `Bytes` | Yes | Transaction hash |

## Orderbook

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| buysQuantity | `BigInt` | Yes | Number of purchases of shares from this order book |
| collateralBuyVolume | `BigInt` | Yes | Global volume of share purchases in USDC base units |
| collateralSellVolume | `BigInt` | Yes | Global volume of share sales in USDC base units |
| collateralVolume | `BigInt` | Yes | Market volume in terms of the underlying collateral value |
| id | `ID` | Yes | Token Id |
| lastActiveDay | `BigInt` | Yes | Timestamp of last day during which someone made a trade |
| scaledCollateralBuyVolume | `BigDecimal` | Yes | Global volume of share purchases in USDC scaled by 10^6 |
| scaledCollateralSellVolume | `BigDecimal` | Yes | Global volume of share sales in USDC scaled by 10^6 |
| scaledCollateralVolume | `BigDecimal` | Yes | Volume scaled by the number of decimals of collateralToken |
| sellsQuantity | `BigInt` | Yes | Number of sales of shares to this order book |
| tradesQuantity | `BigInt` | Yes | Number of trades of any kind against this order book |

## OrdersMatchedEvent

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| id | `ID` | Yes | Transaction Hash |
| makerAmountFilled | `BigInt` | Yes | Maker amount filled |
| makerAssetID | `BigInt` | Yes | Maker asset Id |
| takerAmountFilled | `BigInt` | Yes | Taker amount filled |
| takerAssetID | `BigInt` | Yes | Taker asset Id |
| timestamp | `BigInt` | Yes | Timestamp at which filled occurred |

## OrdersMatchedGlobal

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| buysQuantity | `BigInt` | Yes | Number of purchases of shares from any order book |
| collateralBuyVolume | `BigDecimal` | Yes | Global volume of share purchases in USDC base units |
| collateralSellVolume | `BigDecimal` | Yes | Global volume of share sales in USDC base units |
| collateralVolume | `BigDecimal` | Yes | Global volume in USDC base units |
| id | `ID` | Yes | ID is empty string, this is a singleton |
| scaledCollateralBuyVolume | `BigDecimal` | Yes | Global volume of share purchases in USDC scaled by 10^6 |
| scaledCollateralSellVolume | `BigDecimal` | Yes | Global volume of share sales in USDC scaled by 10^6 |
| scaledCollateralVolume | `BigDecimal` | Yes | Global volume in USDC scaled by 10^6 |
| sellsQuantity | `BigInt` | Yes | Number of sales of shares to any order book |
| tradesQuantity | `BigInt` | Yes | Number of trades of any kind for all order books |

## Query

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| _meta | `_Meta_` | No | Access to subgraph metadata |
| account | `Account` | No |  |
| accounts | `None` | Yes |  |
| collateral | `Collateral` | No |  |
| collaterals | `None` | Yes |  |
| condition | `Condition` | No |  |
| conditions | `None` | Yes |  |
| enrichedOrderFilled | `EnrichedOrderFilled` | No |  |
| enrichedOrderFilleds | `None` | Yes |  |
| fixedProductMarketMaker | `FixedProductMarketMaker` | No |  |
| fixedProductMarketMakers | `None` | Yes |  |
| fpmmFundingAddition | `FpmmFundingAddition` | No |  |
| fpmmFundingAdditions | `None` | Yes |  |
| fpmmFundingRemoval | `FpmmFundingRemoval` | No |  |
| fpmmFundingRemovals | `None` | Yes |  |
| fpmmPoolMembership | `FpmmPoolMembership` | No |  |
| fpmmPoolMemberships | `None` | Yes |  |
| global | `Global` | No |  |
| globals | `None` | Yes |  |
| marketData | `MarketData` | No |  |
| marketDatas | `None` | Yes |  |
| marketPosition | `MarketPosition` | No |  |
| marketPositions | `None` | Yes |  |
| marketProfit | `MarketProfit` | No |  |
| marketProfits | `None` | Yes |  |
| merge | `Merge` | No |  |
| merges | `None` | Yes |  |
| orderFilledEvent | `OrderFilledEvent` | No |  |
| orderFilledEvents | `None` | Yes |  |
| orderbook | `Orderbook` | No |  |
| orderbooks | `None` | Yes |  |
| ordersMatchedEvent | `OrdersMatchedEvent` | No |  |
| ordersMatchedEvents | `None` | Yes |  |
| ordersMatchedGlobal | `OrdersMatchedGlobal` | No |  |
| ordersMatchedGlobals | `None` | Yes |  |
| redemption | `Redemption` | No |  |
| redemptions | `None` | Yes |  |
| split | `Split` | No |  |
| splits | `None` | Yes |  |
| transaction | `Transaction` | No |  |
| transactions | `None` | Yes |  |

## Redemption

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| collateralToken | `Collateral` | Yes | Token which is being claimed in return for outcome tokens |
| condition | `Condition` | Yes | Condition on which redemption is occuring |
| id | `ID` | Yes | Transaction Hash |
| indexSets | `None` | Yes | Outcomes which are being redeemed |
| parentCollectionId | `Bytes` | Yes |  |
| payout | `BigInt` | Yes | The amount of collateral being claimed |
| redeemer | `Account` | Yes | Address which is redeeming these outcomes |
| timestamp | `BigInt` | Yes | Timestamp at which redemption occurred |

## Split

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amount | `BigInt` | Yes | The amount of collateral/outcome tokens being split |
| collateralToken | `Collateral` | Yes | Token which is collateralising positions being split |
| condition | `Condition` | Yes | Condition on which split is occuring |
| id | `ID` | Yes | Transaction Hash |
| parentCollectionId | `Bytes` | Yes |  |
| partition | `None` | Yes |  |
| stakeholder | `Account` | Yes | Address which is performing this split |
| timestamp | `BigInt` | Yes | Timestamp at which split occurred |

## Subscription

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| _meta | `_Meta_` | No | Access to subgraph metadata |
| account | `Account` | No |  |
| accounts | `None` | Yes |  |
| collateral | `Collateral` | No |  |
| collaterals | `None` | Yes |  |
| condition | `Condition` | No |  |
| conditions | `None` | Yes |  |
| enrichedOrderFilled | `EnrichedOrderFilled` | No |  |
| enrichedOrderFilleds | `None` | Yes |  |
| fixedProductMarketMaker | `FixedProductMarketMaker` | No |  |
| fixedProductMarketMakers | `None` | Yes |  |
| fpmmFundingAddition | `FpmmFundingAddition` | No |  |
| fpmmFundingAdditions | `None` | Yes |  |
| fpmmFundingRemoval | `FpmmFundingRemoval` | No |  |
| fpmmFundingRemovals | `None` | Yes |  |
| fpmmPoolMembership | `FpmmPoolMembership` | No |  |
| fpmmPoolMemberships | `None` | Yes |  |
| global | `Global` | No |  |
| globals | `None` | Yes |  |
| marketData | `MarketData` | No |  |
| marketDatas | `None` | Yes |  |
| marketPosition | `MarketPosition` | No |  |
| marketPositions | `None` | Yes |  |
| marketProfit | `MarketProfit` | No |  |
| marketProfits | `None` | Yes |  |
| merge | `Merge` | No |  |
| merges | `None` | Yes |  |
| orderFilledEvent | `OrderFilledEvent` | No |  |
| orderFilledEvents | `None` | Yes |  |
| orderbook | `Orderbook` | No |  |
| orderbooks | `None` | Yes |  |
| ordersMatchedEvent | `OrdersMatchedEvent` | No |  |
| ordersMatchedEvents | `None` | Yes |  |
| ordersMatchedGlobal | `OrdersMatchedGlobal` | No |  |
| ordersMatchedGlobals | `None` | Yes |  |
| redemption | `Redemption` | No |  |
| redemptions | `None` | Yes |  |
| split | `Split` | No |  |
| splits | `None` | Yes |  |
| transaction | `Transaction` | No |  |
| transactions | `None` | Yes |  |

## Transaction

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| feeAmount | `BigInt` | Yes | Amount of collateral paid in fees |
| id | `ID` | Yes | Transaction Hash |
| market | `FixedProductMarketMaker` | Yes | Market which transaction is interacting with |
| outcomeIndex | `BigInt` | Yes | Index of outcome token being bought or sold |
| outcomeTokensAmount | `BigInt` | Yes | Amount of outcome tokens being bought or sold |
| timestamp | `BigInt` | Yes | Timestamp at which transaction occurred |
| tradeAmount | `BigInt` | Yes | Amount of collateral in trade |
| type | `TradeType` | Yes | Buy or Sell transaction |
| user | `Account` | Yes | Account performing transaction |

## _Block_

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| hash | `Bytes` | No | The hash of the block |
| number | `Int` | Yes | The block number |
| parentHash | `Bytes` | No | The hash of the parent block |
| timestamp | `Int` | No | Integer representation of the timestamp stored in blocks for the chain |

## _Meta_

The type for the top-level _meta field

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| block | `_Block_` | Yes | Information about a specific subgraph block. The hash of the block
will be null if the _meta field has a block constraint that asks for
a block number. It will be filled if the _meta field has no block constraint
and therefore asks for the latest  block
 |
| deployment | `String` | Yes | The deployment ID |
| hasIndexingErrors | `Boolean` | Yes | If `true`, the subgraph encountered indexing errors at some past block |

