"""Out-of-time evaluation of 'does Reddit world-news sentiment predict Dow Jones direction?'.
Two framings:
  same_day : headlines of day D -> label of day D  (the Kaggle framing. Reddit posts keep arriving
             after the 4pm ET close, so part of the text may POST-DATE the move: optimistic upper bound)
  next_day : headlines of day D -> label of day D+1 (strict, tradable timing)
Walk-forward: expanding training window, refit every REFIT days, test only on later dates.
Uncertainty: moving-block bootstrap CI on the accuracy gain over always-up, and an exact binomial test."""
import json, sqlite3, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import *
START, REFIT, BLOCK = 1000, 63, 20
SENT = ["s_mean", "s_min", "s_max", "s_neg", "s_pos"]

def daily(db=DB_PATH):
    con = sqlite3.connect(db); d = pd.read_sql("select * from days", con)
    h = pd.read_sql("select * from scored_headlines", con); g = h.groupby("date")
    f = pd.DataFrame({"s_mean": g.compound.mean(), "s_min": g.compound.min(), "s_max": g.compound.max(),
                      "s_neg": g.compound.apply(lambda s: (s < -.05).mean()), "s_pos": g.compound.apply(lambda s: (s > .05).mean()),
                      "text": g.text.apply(" . ".join)})
    return d.merge(f, left_on="date", right_index=True).sort_values("date").reset_index(drop=True)

def frame(d, mode):
    d = d.copy(); d["y"] = d.label if mode == "same_day" else d.label.shift(-1)
    d["prev_label"] = d.label.shift(1) if mode == "same_day" else d.label       # label known before prediction
    return d.dropna(subset=["y", "prev_label"]).reset_index(drop=True).astype({"y": int})

def walk_forward(d, start=START, refit=REFIT):
    preds = {k: np.full(len(d), np.nan) for k in ["always_up", "prev_day_direction", "vader_logreg", "tfidf_logreg"]}
    for i in range(start, len(d), refit):
        tr, te = d.iloc[:i], d.iloc[i:i + refit]; assert tr.date.max() < te.date.min()   # no look-ahead
        preds["always_up"][i:i + refit] = float(tr.y.mean() >= .5)
        preds["prev_day_direction"][i:i + refit] = te.prev_label.values
        m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000)).fit(tr[SENT], tr.y)
        preds["vader_logreg"][i:i + refit] = m.predict_proba(te[SENT])[:, 1]
        t = make_pipeline(TfidfVectorizer(min_df=5, max_features=20000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
                          LogisticRegression(C=.1, max_iter=1000)).fit(tr.text, tr.y)
        preds["tfidf_logreg"][i:i + refit] = t.predict_proba(te.text)[:, 1]
    return preds

def block_bootstrap_ci(diff, block=BLOCK, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); L = len(diff); nb = int(np.ceil(L / block)); out = []
    for _ in range(n):
        st = rng.integers(0, L - block + 1, nb); out.append(np.concatenate([diff[s:s + block] for s in st])[:L].mean())
    return [round(float(np.percentile(out, q)), 4) for q in (2.5, 97.5)]

def evaluate(d, mode):
    f = frame(d, mode); preds = walk_forward(f); ev = f.iloc[START:].copy(); y = ev.y.values; rows = []
    base_correct = (preds["always_up"][START:] == y).astype(float)
    for name, p in preds.items():
        p = p[START:]; hard = (p >= .5).astype(int); correct = (hard == y).astype(float); acc = correct.mean()
        diff = correct - base_correct
        rows.append({"framing": mode, "model": name, "n_test_days": len(y), "accuracy": round(acc, 4),
                     "auc": round(roc_auc_score(y, p), 4) if name in ("vader_logreg", "tfidf_logreg") else np.nan,
                     "gain_vs_always_up": round(diff.mean(), 4),
                     "gain_95ci": block_bootstrap_ci(diff) if name != "always_up" else None,
                     "binom_p_vs_always_up_rate": round(binomtest(int(correct.sum()), len(y), base_correct.mean(), alternative="greater").pvalue, 4) if name != "always_up" else None})
    return rows, f

def top_words(f, k=12):
    v = TfidfVectorizer(min_df=5, ngram_range=(1, 1), stop_words="english", sublinear_tf=True); X = v.fit_transform(f.text)
    c = LogisticRegression(C=.1, max_iter=1000).fit(X, f.y).coef_[0]; w = np.array(v.get_feature_names_out()); o = np.argsort(c)
    return pd.DataFrame({"word": list(w[o[-k:]][::-1]) + list(w[o[:k]]), "direction": ["up"] * k + ["down"] * k,
                         "coef": list(c[o[-k:]][::-1]) + list(c[o[:k]])}).round(3)

def main(db=DB_PATH, out=REPORTS):
    d = daily(db); all_rows = []
    for mode in ("same_day", "next_day"):
        rows, f = evaluate(d, mode); all_rows += rows
    r = pd.DataFrame(all_rows); r.to_csv(out / "results.csv", index=False); print(r.to_string(index=False))
    top_words(frame(d, "next_day")).to_csv(out / "top_words_next_day.csv", index=False)
    daily_out = d[["date", "label", "s_mean"]]; daily_out.to_csv(out / "daily_sentiment.csv", index=False)
    return r

if __name__ == "__main__":
    main()
