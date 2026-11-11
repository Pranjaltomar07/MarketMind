"""Stronger tests of the null result, using only the daily table in reports/daily_sentiment.csv
(date, label, s_mean), so it runs without the raw headlines or any download:  python src/signal_tests.py

The original study found nothing with headline-level VADER features. A fair reader asks three things,
answered here:
  1. Did you only try one kind of feature?   Seven feature sets: sentiment level, sentiment SHOCK (today
     against its own recent average), 1-day change, 5/20-day averages, weekday, label momentum, and all together.
  2. Could chance alone explain a lucky or unlucky result?   A circular-shift permutation test for AUC
     (keeps the autocorrelation of both series) with Holm correction across the seven looks, plus block-bootstrap CIs.
  3. Could the test have seen a real signal at all?   A power study: plant a signal of known strength in the real
     sentiment series, run the same walk-forward code, and count how often it is detected.
Every model is walk-forward: refit every 63 days on the past only, predict the next day's direction.
"""
import numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import REPORTS, SEED
from evaluate import block_bootstrap_ci, START, REFIT, BLOCK

SETS = {"sentiment_level": ["s_mean"], "sentiment_shock_20d": ["shock20"], "sentiment_change_1d": ["d1"],
        "averages_5d_20d": ["m5", "m20"], "weekday": ["dow1", "dow2", "dow3", "dow4"], "label_momentum_20d": ["mom20"],
        "all_combined": ["s_mean", "shock20", "d1", "m5", "m20", "dow1", "dow2", "dow3", "dow4", "mom20"]}


def load(path=REPORTS / "daily_sentiment.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Every feature at row t uses information available at the close of day t. Target y = direction of day t+1."""
    f = d.copy(); s = f.s_mean
    f["shock20"] = (s - s.shift(1).rolling(20, min_periods=5).mean()).fillna(0)
    f["d1"] = (s - s.shift(1)).fillna(0); f["m5"] = s.rolling(5, min_periods=1).mean(); f["m20"] = s.rolling(20, min_periods=1).mean()
    f["mom20"] = f.label.rolling(20, min_periods=1).mean()
    nxt = f.date.shift(-1).dt.dayofweek.fillna(0).astype(int)                    # weekday of the day being predicted (known in advance)
    for k in (1, 2, 3, 4): f[f"dow{k}"] = (nxt == k).astype(int)
    f["y"] = f.label.shift(-1); return f.dropna(subset=["y"]).reset_index(drop=True).astype({"y": int})


def walk_forward(f: pd.DataFrame, cols, start=START, refit=REFIT) -> np.ndarray:
    p = np.full(len(f), np.nan)
    for i in range(start, len(f), refit):
        tr, te = f.iloc[:i], f.iloc[i:i + refit]; assert tr.date.max() < te.date.min()      # no look-ahead
        m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000)).fit(tr[cols], tr.y); p[i:i + refit] = m.predict_proba(te[cols])[:, 1]
    return p[start:]


def boot_auc_ci(y, p, block=BLOCK, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); L = len(y); nb = int(np.ceil(L / block)); out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, L - block + 1, nb)])[:L]
        if y[idx].min() != y[idx].max(): out.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(out, q)), 4) for q in (2.5, 97.5)]


def shift_perm_p(y, p, n=2000, min_shift=25, seed=SEED) -> float:
    """Two-sided p for AUC: compare with AUCs after circularly shifting the predictions against the labels.
    Shifting keeps each series' own autocorrelation, which a plain shuffle would destroy."""
    rng = np.random.default_rng(seed); obs = abs(roc_auc_score(y, p) - .5); L = len(y)
    null = np.array([abs(roc_auc_score(y, np.roll(p, int(k))) - .5) for k in rng.integers(min_shift, L - min_shift, n)])
    return float((1 + (null >= obs).sum()) / (n + 1))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items): run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def run_sets(f: pd.DataFrame) -> pd.DataFrame:
    base_p = np.full(len(f) - START, float(f.y.iloc[:START].mean() >= .5)); y = f.y.values[START:]
    base_correct = (base_p == y).astype(float); rows = []
    for name, cols in SETS.items():
        p = walk_forward(f, cols); correct = ((p >= .5).astype(int) == y).astype(float)
        rows.append({"feature_set": name, "n_test_days": len(y), "accuracy": round(correct.mean(), 4), "auc": round(roc_auc_score(y, p), 4),
                     "auc_95ci": boot_auc_ci(y, p), "gain_vs_always_up": round((correct - base_correct).mean(), 4),
                     "gain_95ci": block_bootstrap_ci(correct - base_correct), "perm_p": round(shift_perm_p(y, p), 4)})
    r = pd.DataFrame(rows); adj = holm(dict(zip(r.feature_set, r.perm_p))); r["holm_p"] = r.feature_set.map(adj).round(4); return r


def subperiods(f: pd.DataFrame, k=3) -> pd.DataFrame:
    y = f.y.values[START:]; dates = f.date.values[START:]; edges = np.linspace(0, len(y), k + 1).astype(int); rows = []
    for name in ("sentiment_level", "sentiment_shock_20d", "averages_5d_20d"):
        p = walk_forward(f, SETS[name])
        for a, b in zip(edges[:-1], edges[1:]):
            rows.append({"feature_set": name, "from": str(dates[a])[:10], "to": str(dates[b - 1])[:10], "n": b - a, "auc": round(roc_auc_score(y[a:b], p[a:b]), 4)})
    return pd.DataFrame(rows)


def null_se(y) -> float:
    """Standard error of AUC under no signal (Hanley-McNeil null form)."""
    n1 = int(y.sum()); n0 = len(y) - n1; return float(np.sqrt((n0 + n1 + 1) / (12 * n0 * n1)))


def power_study(f: pd.DataFrame, betas=(0, .1, .2, .3, .45, .6), sims=120, seed=SEED) -> pd.DataFrame:
    """Plant logit(P(up tomorrow)) = a + beta * z(sentiment today) in the REAL sentiment series, run the same
    walk-forward pipeline, and count one-sided 5% detections (AUC above 0.5 by more than 1.645 null standard errors)."""
    rng = np.random.default_rng(seed); z = ((f.s_mean - f.s_mean.mean()) / f.s_mean.std()).values; a = float(logit(.535)); rows = []
    for b in betas:
        aucs, hits = [], []
        for _ in range(sims):
            g = f.copy(); g["y"] = (rng.random(len(g)) < expit(a + b * z)).astype(int); y = g.y.values[START:]
            p = walk_forward(g, ["s_mean"]); auc = roc_auc_score(y, p); aucs.append(auc); hits.append(auc > .5 + 1.645 * null_se(y))
        rows.append({"planted_beta": b, "mean_auc": round(float(np.mean(aucs)), 4), "detection_rate": round(float(np.mean(hits)), 3)})
    return pd.DataFrame(rows)


def main():
    f = features(load()); REPORTS.mkdir(exist_ok=True)
    r = run_sets(f); r.to_csv(REPORTS / "signal_tests.csv", index=False); print(r.to_string(index=False))
    s = subperiods(f); s.to_csv(REPORTS / "signal_subperiods.csv", index=False); print(s.to_string(index=False))
    y = f.y.values[START:]; se = null_se(y); mde = round(.5 + 1.645 * se, 4); p80 = round(.5 + (1.645 + .8416) * se, 4)
    print(f"\nnull SE of AUC with {len(y)} test days: {se:.4f}; AUC needed to detect: {mde}; for 80% power: {p80}")
    pw = power_study(f); pw.to_csv(REPORTS / "signal_power.csv", index=False); print(pw.to_string(index=False))
    pd.Series({"n_test_days": len(y), "null_se_auc": round(se, 4), "auc_detectable_5pct": mde, "auc_for_80pct_power": p80}).to_json(REPORTS / "signal_power_summary.json")


if __name__ == "__main__":
    main()
