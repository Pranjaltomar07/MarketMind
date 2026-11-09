# MarketMind - News Sentiment vs. Stock Market Direction

[![tests](https://github.com/YOUR-USERNAME/marketmind/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranjaltomar07/marketmind/actions)

*Does news sentiment predict Dow Jones direction? A leakage-safe test with control experiments.*

I wanted to test a popular claim properly: *can the sentiment of daily world-news headlines predict the direction of the Dow Jones?* The data is real: the top 25 Reddit r/worldnews headlines per day with Dow Jones up/down labels, 2008 to 2016 (1,989 days, 49,718 headlines).

**Result: no.** Under strict out-of-time evaluation no model beats "always predict up" (53.6% of days). What I took from it is the method: walk-forward testing without look-ahead, control tests that show the framework can detect a signal when one exists, and confidence intervals instead of a single accuracy number. See [FINDINGS.md](FINDINGS.md).

| Framing (988 test days) | Model | Accuracy | Gain vs always-up (95% block-bootstrap CI) |
|---|---|---|---|
| next_day (strict) | always-up baseline | 53.6% | - |
| next_day | VADER sentiment + logistic regression | 54.5% | +0.8% [-0.7%, +2.2%] |
| next_day | TF-IDF + logistic regression | 53.6% | 0.0% (collapses to majority class; AUC 0.50) |
| same_day (optimistic) | VADER sentiment + logistic regression | 52.2% | -1.4% [-3.0%, +0.2%] |

## Run
```bash
pip install -r requirements.txt
python run_all.py            # download -> ETL -> streaming sentiment scoring -> walk-forward evaluation
streamlit run app.py
pytest -q tests              # offline; includes positive/negative control tests
docker build -t marketmind . && docker run -p 8501:8501 marketmind
```

## Pipeline
| Stage | File |
|---|---|
| Ingest | `src/download.py` (public CSV, not redistributed), `src/etl.py` (decodes byte-string headlines, wide -> long, SQLite) |
| Stream scoring | `src/ingest.py`: bounded queue + 4 worker threads, VADER + small finance lexicon, measured throughput (~19k headlines/s in-process; a Kafka topic would replace the queue) |
| Evaluation | `src/evaluate.py`: expanding-window walk-forward (refit every 63 days, test on 988 later days), two timing framings, block bootstrap CIs, exact binomial test |

## Why I trust the null result
- **No look-ahead:** each fold asserts `max(train date) < min(test date)`; next-day framing shifts labels forward.
- **Positive control** (`tests/`): with a planted sentiment->direction link the same code recovers a >10-point accuracy gain.
- **Negative control:** with no planted link it finds no gain.
- Gains are measured against the right baseline (always-up), not 50%, and uncertainty is reported.

## Limitations
VADER is a general-purpose lexicon, and a finance-specific model such as FinBERT would be the next step. Daily aggregation discards intraday timing. The dataset has labels but no prices, so I did not attempt a return-based backtest.

## Results at a glance
![Gain vs baseline](docs/img/gain_vs_baseline.png)
![Sentiment quintiles](docs/img/sentiment_quintiles.png)

## How this was built
Developed with AI assistance and personal coding and implemetation. Numbers in this README come from running the code in this repository.

## Author
Pranjal Tomar. Part of my [portfolio](https://github.com/Pranjaltomar07).
