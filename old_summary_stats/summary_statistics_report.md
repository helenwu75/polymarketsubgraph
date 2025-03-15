# Polymarket Election Markets Analysis: Summary Statistics

*Generated on: 2025-03-14 22:55:02*

## Dataset Overview

- **Total Markets:** 1048575
- **Total Features:** 54

## Target Variable Analysis

### Prediction Correctness Distribution

- **Correct Predictions:** 461 (94.5%)
- **Incorrect Predictions:** 27 (5.5%)

- **Class Imbalance Ratio:** 17.07

![Target Distribution](target_distribution.png)

## Prediction Quality Metrics

### Summary of Pre-calculated Metrics

- **Brier Score (Mean):** 0.0372
- **Brier Score (Median):** 0.0000

![Brier Score Distribution](brier_score_distribution.png)

- **Log Loss (Mean):** 0.1216
- **Log Loss (Median):** 0.0030

![Log Loss Distribution](log_loss_distribution.png)

### Correlation Between Metrics

![Prediction Metrics Correlation](prediction_metrics_correlation.png)

## Correlations with Prediction Correctness

![Correlations with Target](target_correlation.png)

## Election Type Analysis

![Accuracy by Election Type](accuracy_by_event_electionType.png)

## Feature Distributions

![Feature Distributions](feature_distributions.png)

## Recommendations for Metric Selection

Based on the analysis of the pre-calculated metrics, we recommend:

1. **Use Brier Score as primary metric** - Provides a balanced assessment of prediction accuracy
2. **Use Log Loss as supporting metric** - Penalizes confident but wrong predictions
3. **Use prediction correctness for simplicity** - When a binary measure is needed for interpretability

## Summary

The markets show an overall prediction accuracy of 94.5%. The analysis identified key factors associated with prediction accuracy that can help understand which features contribute to successful predictions in election markets.
