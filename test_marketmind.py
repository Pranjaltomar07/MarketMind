import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import etl, ingest, evaluate as ev

def synth_days(n=1500, planted=0.0, seed=0):
    """Synthetic daily table. `planted` = strength of the sentiment -> next-day-direction link."""
    rng = np.random.default_rng(seed); s = rng.normal(0, 1, n)
    y_next = (rng.random(n) < 1 / (1 + np.exp(-(0.15 + planted * s)))).astype(int)   # y_next[t] = label of day t+1
    label = np.r_[1, y_next[:-1]]                                                      # label[t] aligned to its own day
    return pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n), "label": label, "s_mean": s, "s_min": s - 1,
                         "s_max": s + 1, "s_neg": 1 / (1 + np.exp(s)), "s_pos": 1 / (1 + np.exp(-s)),
                         "text": ["word%d news" % (int(v * 3) % 7) for v in s]})

def test_clean_text_decodes_byte_literals():
    assert etl.clean_text("b'Georgia downs jets'") == "Georgia downs jets"
    assert etl.clean_text('b"It\'s fine"') == "It's fine"
    assert etl.clean_text(float("nan")) == "" and etl.clean_text("plain") == "plain"

def test_etl_long_format(tmp_path):
    csv = tmp_path / "c.csv"
    pd.DataFrame({"Date": ["02/01/2010", "01/01/2010"], "Label": [1, 0], "Top1": ["b'a'", "b'b'"], "Top2": ["b'c'", None]}).to_csv(csv, index=False)
    days, long = etl.build(csv)
    assert list(days.date.dt.day) == [1, 2] and len(long) == 3          # sorted, day-first parsed, empty headline dropped

def test_streaming_scores_every_item_once():
    rows = [{"text": t, "i": i} for i, t in enumerate(["stocks surge on strong growth", "market crash fears", "meeting held"] * 200)]
    out = ingest.score_stream(rows, workers=4)
    assert sorted(r["i"] for r, _ in out) == list(range(600))
    pos = [c for r, c in out if "surge" in r["text"]][0]; neg = [c for r, c in out if "crash" in r["text"]][0]
    assert pos > 0 > neg

def test_next_day_framing_has_no_lookahead():
    d = synth_days(50); f = ev.frame(d, "next_day")
    assert (f.y.values == d.label.shift(-1).dropna().values).all() and len(f) == 49

def test_walk_forward_trains_only_on_the_past():
    d = ev.frame(synth_days(1200), "same_day")
    ev.walk_forward(d, start=1000, refit=63)     # the function asserts max(train date) < min(test date) every fold

def test_negative_control_no_signal_no_gain():
    f = ev.frame(synth_days(1500, planted=0.0), "next_day"); p = ev.walk_forward(f)
    y = f.y.values[ev.START:]; base = (p["always_up"][ev.START:] == y).mean(); m = ((p["vader_logreg"][ev.START:] >= .5) == y).mean()
    assert m - base < 0.05

def test_positive_control_detects_planted_signal():
    f = ev.frame(synth_days(1500, planted=1.5), "next_day"); p = ev.walk_forward(f)
    y = f.y.values[ev.START:]; base = (p["always_up"][ev.START:] == y).mean(); m = ((p["vader_logreg"][ev.START:] >= .5) == y).mean()
    assert m - base > 0.10, "framework failed to recover a strong planted signal"

def test_block_bootstrap_ci_brackets_true_mean():
    diff = np.random.default_rng(1).choice([-1., 0., 1.], 900, p=[.2, .6, .2]); lo, hi = ev.block_bootstrap_ci(diff, n=500)
    assert lo < diff.mean() < hi
