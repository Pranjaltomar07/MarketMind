"""Streaming-style ingestion + sentiment scoring: a producer pushes headlines into a bounded queue
(stand-in for a Kafka topic), worker threads score them with VADER + a small finance lexicon.
Throughput is measured (in-process; NOT a Kafka benchmark) and saved to reports/ingest_stats.json."""
import queue, threading, time, json, sqlite3, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import *

FIN_LEX = {"crash": -3.0, "plunge": -2.8, "recession": -2.5, "default": -2.2, "bankruptcy": -3.0, "sanctions": -1.8,
           "rally": 2.2, "surge": 2.4, "recovery": 1.8, "growth": 1.5, "bailout": -1.2, "deficit": -1.5}

def make_analyzer():
    s = SentimentIntensityAnalyzer(); s.lexicon.update(FIN_LEX); return s

def score_stream(rows, workers=4, analyzer=None):
    """rows: list of dicts with 'text'. Returns list of (row, compound) preserving all rows."""
    sia = analyzer or make_analyzer(); q = queue.Queue(maxsize=5000); out, lock = [], threading.Lock()
    def produce():
        for r in rows: q.put(r)
        for _ in range(workers): q.put(None)
    def consume():
        loc = []
        while (r := q.get()) is not None: loc.append((r, sia.polarity_scores(r["text"])["compound"]))
        with lock: out.extend(loc)
    ts = [threading.Thread(target=consume) for _ in range(workers)]; [t.start() for t in ts]
    threading.Thread(target=produce).start(); [t.join() for t in ts]; return out

def main():
    con = sqlite3.connect(DB_PATH); rows = pd.read_sql("select * from headlines", con).to_dict("records")
    t0 = time.time(); res = score_stream(rows); dt = time.time() - t0
    df = pd.DataFrame([{**r, "compound": c} for r, c in res]); df.to_sql("scored_headlines", con, if_exists="replace", index=False)
    st = {"headlines": len(df), "seconds": round(dt, 2), "headlines_per_sec": round(len(df) / dt)}
    (REPORTS / "ingest_stats.json").write_text(json.dumps(st, indent=2)); print(st)

if __name__ == "__main__":
    main()
