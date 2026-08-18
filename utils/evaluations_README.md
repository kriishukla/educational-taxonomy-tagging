# LLM Rectification Evaluation Metrics

This document explains the evaluation metrics used to measure **LLM
rectification quality**. Rectification refers to improving an initial
LLM score so that it better aligns with **gold (ground‑truth) values**.

------------------------------------------------------------------------

## Data Assumptions

For each score prefix (e.g., `Subject_validity`), the dataframe
contains:

-   `<prefix>_llm` --- Original LLM prediction
-   `<prefix>_rectified` --- Corrected prediction
-   `<prefix>_gold` --- Ground truth score
-   `<prefix>_ci_size` --- Confidence interval size (uncertainty)

------------------------------------------------------------------------

## Core Evaluation Goal

We measure whether rectification:

1.  Reduces prediction error
2.  Improves ranking agreement
3.  Moves predictions toward ground truth
4.  Uses uncertainty meaningfully

------------------------------------------------------------------------

## Metrics Explained

### 1. Mean Absolute Error (MAE)

Measures average absolute difference between prediction and gold.

Formula:

MAE = mean(\|prediction − gold\|)

Interpretation: - Lower is better - Primary accuracy metric

Rectification success: MAE_rectified \< MAE_llm

------------------------------------------------------------------------

### 2. Root Mean Squared Error (RMSE)

Penalizes large errors more heavily.

Formula:

RMSE = sqrt(mean((prediction − gold)\^2))

Use when large mistakes matter.

------------------------------------------------------------------------

### 3. Spearman Correlation

Measures ranking agreement between predictions and gold values.

Why important: Many LLM scores are relative rather than absolute.

Higher correlation = better ordering consistency.

------------------------------------------------------------------------

### 4. Improvement Rate

Percentage of samples where rectified prediction is closer to gold.

Formula:

Improvement Rate = mean(\|rectified − gold\| \< \|llm − gold\|)

Interpretation: Directly shows how often rectification helps.

------------------------------------------------------------------------

### 5. Error Reduction Percentage

Measures proportional improvement.

Formula:

Error Reduction = (MAE_llm − MAE_rectified) / MAE_llm

Example: 20% means rectification removed one‑fifth of error.

------------------------------------------------------------------------

### 6. Directional Correctness

Checks whether rectification moved prediction in the correct direction.

If gold \> llm → rectified should increase. If gold \< llm → rectified
should decrease.

Measures correction quality, not just magnitude.

------------------------------------------------------------------------

### 7. Confidence‑Weighted Error

Uses confidence interval size to evaluate uncertainty calibration.

Formula:

\|rectified − gold\| / CI_size

Desired behavior: - Small error when confident - Larger uncertainty when
unsure

Lower values indicate better uncertainty usage.

------------------------------------------------------------------------

## Recommended Metric Set

For robust evaluation:

-   MAE (primary)
-   RMSE
-   Spearman correlation gain
-   Improvement rate
-   Error reduction %
-   Directional correctness
-   Confidence‑weighted error

------------------------------------------------------------------------

## Output Interpretation

Typical evaluation table:

  Metric             LLM   Rectified   Delta
  ------------------ ----- ----------- ----------------
  MAE                ↓     ↓↓          Improvement
  RMSE               ↓     ↓↓          Improvement
  Spearman           ↑     ↑↑          Better ranking
  Improvement Rate   ---   High        Good
  Error Reduction    ---   Positive    Good

------------------------------------------------------------------------

## Best Practices

-   Always evaluate against **gold**, not LLM alone.
-   Report both absolute error and ranking metrics.
-   Use multiple score prefixes independently.
-   Aggregate results for comparison across experiments.

------------------------------------------------------------------------

## Summary

Rectification quality is multi‑dimensional:

-   Accuracy (MAE, RMSE)
-   Ranking alignment (Spearman)
-   Correction behavior (Improvement Rate, Directional Correctness)
-   Calibration (Confidence‑Weighted Error)

Together, these metrics provide a comprehensive evaluation of LLM
rectification systems.
