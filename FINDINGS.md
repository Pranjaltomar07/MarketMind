# Findings - MarketMind (news sentiment vs Dow Jones direction)

## Question
Does the sentiment or content of the day's top world-news headlines carry information about whether the Dow Jones goes up?

## Data
1,989 trading days (2008-08-08 to 2016-07-01); 25 headlines per day; label = 1 if the Dow closed up or flat (53.5% of days). Mean headline sentiment is negative (-0.22): world news is mostly bad news and varies little (std 0.09).

## Method
Expanding-window walk-forward: first 1,000 days train, then refit every 63 days and predict the next block (988 test days). Models: always-up, yesterday's direction, VADER-feature logistic regression, TF-IDF (1-2 grams) logistic regression. Two timings: same-day (Kaggle framing, optimistic because Reddit posts can post-date the close) and next-day (strict).

## Results
- Nothing reliably beats always-up (53.6%). Best point estimate: VADER next-day 54.5%, gain +0.8 points, 95% CI [-0.7, +2.2], exact binomial p = 0.32.
- TF-IDF learns to predict the majority class (AUC 0.50). Predicting "same as yesterday" is significantly worse than always-up (-6.5 points).
- Same-day framing, which has the most favourable timing, is not better (AUC 0.48), so the optimistic timing does not rescue the signal.
- Up-rate by sentiment quintile is flat and non-monotone (50-56%).
- Words with the largest coefficients (merkel, australia, earthquake, malaysia) are noise fits with no plausible link to the market.

## Why this matters
The same code detects a planted signal (positive control test) and finds nothing when none exists (negative control), so the null result reflects the data, not a broken pipeline. Six model/framing comparisons were run; with that many looks, even a "significant" result at p<0.05 would need caution.

## Limitations and next steps
General-purpose sentiment lexicon rather than finance-specific (FinBERT would be the next step, with network access); daily granularity; no prices/returns so no trading backtest; one equity index. Next: intraday timestamps, per-company news, and return-based evaluation with transaction costs.
