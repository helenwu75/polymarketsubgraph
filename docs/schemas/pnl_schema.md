# PNL Subgraph Schema Documentation

Generated on: 2025-02-12 21:33:53

## Condition

### Fields

| Field Name        | Type     | Required | Description |
| ----------------- | -------- | -------- | ----------- |
| id                | `ID`     | Yes      | conditionId |
| payoutDenominator | `BigInt` | Yes      |             |
| payoutNumerators  | `None`   | Yes      | payouts     |
| positionIds       | `None`   | Yes      | tokenIds    |

## FPMM

### Fields

| Field Name  | Type     | Required | Description  |
| ----------- | -------- | -------- | ------------ |
| conditionId | `String` | Yes      | conditionId  |
| id          | `ID`     | Yes      | FPMM address |

## NegRiskEvent

### Fields

| Field Name    | Type  | Required | Description     |
| ------------- | ----- | -------- | --------------- |
| id            | `ID`  | Yes      | negRiskMarketId |
| questionCount | `Int` | Yes      | Question Count  |

## Query

### Fields

| Field Name    | Type           | Required | Description                 |
| ------------- | -------------- | -------- | --------------------------- |
| \_meta        | `_Meta_`       | No       | Access to subgraph metadata |
| condition     | `Condition`    | No       |                             |
| conditions    | `None`         | Yes      |                             |
| fpmm          | `FPMM`         | No       |                             |
| fpmms         | `None`         | Yes      |                             |
| negRiskEvent  | `NegRiskEvent` | No       |                             |
| negRiskEvents | `None`         | Yes      |                             |
| userPosition  | `UserPosition` | No       |                             |
| userPositions | `None`         | Yes      |                             |

## Subscription

### Fields

| Field Name    | Type           | Required | Description                 |
| ------------- | -------------- | -------- | --------------------------- |
| \_meta        | `_Meta_`       | No       | Access to subgraph metadata |
| condition     | `Condition`    | No       |                             |
| conditions    | `None`         | Yes      |                             |
| fpmm          | `FPMM`         | No       |                             |
| fpmms         | `None`         | Yes      |                             |
| negRiskEvent  | `NegRiskEvent` | No       |                             |
| negRiskEvents | `None`         | Yes      |                             |
| userPosition  | `UserPosition` | No       |                             |
| userPositions | `None`         | Yes      |                             |

## UserPosition

### Fields

| Field Name  | Type     | Required | Description                             |
| ----------- | -------- | -------- | --------------------------------------- |
| amount      | `BigInt` | Yes      | amount of token the user holds          |
| avgPrice    | `BigInt` | Yes      | the avg price the user bought the token |
| id          | `ID`     | Yes      | User Address + Token ID                 |
| realizedPnl | `BigInt` | Yes      | realized profits - losses               |
| tokenId     | `BigInt` | Yes      | Token ID                                |
| totalBought | `BigInt` | Yes      | total amount of token bought            |
| user        | `String` | Yes      | User Address                            |

## _Block_

### Fields

| Field Name | Type    | Required | Description                                                            |
| ---------- | ------- | -------- | ---------------------------------------------------------------------- |
| hash       | `Bytes` | No       | The hash of the block                                                  |
| number     | `Int`   | Yes      | The block number                                                       |
| parentHash | `Bytes` | No       | The hash of the parent block                                           |
| timestamp  | `Int`   | No       | Integer representation of the timestamp stored in blocks for the chain |

## _Meta_

The type for the top-level \_meta field

### Fields

| Field Name | Type      | Required | Description                                                        |
| ---------- | --------- | -------- | ------------------------------------------------------------------ |
| block      | `_Block_` | Yes      | Information about a specific subgraph block. The hash of the block |

will be null if the \_meta field has a block constraint that asks for
a block number. It will be filled if the \_meta field has no block constraint
and therefore asks for the latest block
|
| deployment | `String` | Yes | The deployment ID |
| hasIndexingErrors | `Boolean` | Yes | If `true`, the subgraph encountered indexing errors at some past block |
