# Activity Subgraph Schema Documentation

Generated on: 2025-02-12 21:33:53

## Condition

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| id | `ID` | Yes | Condition ID |

## FixedProductMarketMaker

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| id | `ID` | Yes | Market maker address |

## Merge

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amount | `BigInt` | Yes | The amount of collateral/outcome tokens being merged |
| condition | `String` | Yes | Token which is collateralising positions being merged |
| id | `ID` | Yes | Transaction Hash |
| stakeholder | `String` | Yes | Address which is performing this merge |
| timestamp | `BigInt` | Yes | Timestamp at which merge occurred |

## NegRiskConversion

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amount | `BigInt` | Yes | The amount of each token being converted |
| id | `ID` | Yes | Transaction Hash |
| indexSet | `BigInt` | Yes | The index set of the outcome tokens being converted |
| negRiskMarketId | `String` | Yes | Neg Risk Market Id assigned to the event |
| questionCount | `Int` | Yes | The number of questions at the time of conversion |
| stakeholder | `String` | Yes | Address which is performing this conversion |
| timestamp | `BigInt` | Yes | Timestamp at which conversion occurred |

## NegRiskEvent

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| id | `ID` | Yes | negRiskMarketId |
| questionCount | `Int` | Yes | Question Count |

## Position

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| condition | `String` | Yes | Condition that the token is linked to |
| id | `ID` | Yes | ERC1155 TokenID of the CTF Asset |
| outcomeIndex | `BigInt` | Yes | Outcome Index |

## Query

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| _meta | `_Meta_` | No | Access to subgraph metadata |
| condition | `Condition` | No |  |
| conditions | `None` | Yes |  |
| fixedProductMarketMaker | `FixedProductMarketMaker` | No |  |
| fixedProductMarketMakers | `None` | Yes |  |
| merge | `Merge` | No |  |
| merges | `None` | Yes |  |
| negRiskConversion | `NegRiskConversion` | No |  |
| negRiskConversions | `None` | Yes |  |
| negRiskEvent | `NegRiskEvent` | No |  |
| negRiskEvents | `None` | Yes |  |
| position | `Position` | No |  |
| positions | `None` | Yes |  |
| redemption | `Redemption` | No |  |
| redemptions | `None` | Yes |  |
| split | `Split` | No |  |
| splits | `None` | Yes |  |

## Redemption

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| condition | `String` | Yes | Condition on which redemption is occuring |
| id | `ID` | Yes | Transaction Hash |
| indexSets | `None` | Yes | Outcomes which are being redeemed |
| payout | `BigInt` | Yes | The amount of collateral being claimed |
| redeemer | `String` | Yes | Address which is redeeming these outcomes |
| timestamp | `BigInt` | Yes | Timestamp at which redemption occurred |

## Split

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| amount | `BigInt` | Yes | The amount of collateral/outcome tokens being split |
| condition | `String` | Yes | Condition on which split is occuring |
| id | `ID` | Yes | Transaction Hash |
| stakeholder | `String` | Yes | Address which is performing this split |
| timestamp | `BigInt` | Yes | Timestamp at which split occurred |

## Subscription

### Fields

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| _meta | `_Meta_` | No | Access to subgraph metadata |
| condition | `Condition` | No |  |
| conditions | `None` | Yes |  |
| fixedProductMarketMaker | `FixedProductMarketMaker` | No |  |
| fixedProductMarketMakers | `None` | Yes |  |
| merge | `Merge` | No |  |
| merges | `None` | Yes |  |
| negRiskConversion | `NegRiskConversion` | No |  |
| negRiskConversions | `None` | Yes |  |
| negRiskEvent | `NegRiskEvent` | No |  |
| negRiskEvents | `None` | Yes |  |
| position | `Position` | No |  |
| positions | `None` | Yes |  |
| redemption | `Redemption` | No |  |
| redemptions | `None` | Yes |  |
| split | `Split` | No |  |
| splits | `None` | Yes |  |

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

