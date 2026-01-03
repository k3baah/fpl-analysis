# FPL Price Chasing Analysis

**Question**: When players have price rises in FPL, do they earn more points the following week? How bad is "chasing price rises"?

## The Problem

Many FPL managers buy players after their price rises, assuming recent form will continue. This analysis investigates whether this "price chasing" strategy actually works.

## Data Source

Uses historical FPL data from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repository.

## Quick Start

```bash
pip install -r requirements.txt
python analyze_price_chasing.py
```

## What It Analyzes

1. **Price Rise → Next Week Points**: Do players who rise in price score more points the following gameweek?
2. **Price Fall → Next Week Points**: Do falling players underperform?
3. **Transfer Activity**: How do transfers correlate with subsequent performance?
4. **Regression to Mean**: Do high-scoring players regress in following weeks?

## Key Metrics

- Correlation between price change and next-GW points
- Average points by price change category (risers/stable/fallers)
- Form persistence analysis
