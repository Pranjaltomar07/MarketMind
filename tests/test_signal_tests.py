import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import signal_tests as st


def daily(n=1500, beta=0.0, seed=0):
    r = np.random.default_rng(seed); s = r.normal(-.2, .09, n); z = (s - s.mean()) / s.std()
    y_next = (r.random(n) < 1 / (1 + np.exp(-(.14 + beta * z)))).astype(int); label = np.r_[1, y_next[:-1]]
    return pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n), "label": label, "s_mean": s})


def test_target_is_next_day_label():
    d = daily(60); f = st.features(d); assert (f.y.values == d.label.shift(-1).dropna().values).all() and len(f) == 59
def test_features_do_not_use_the_future():
    d = daily(80); a = st.features(d); d2 = d.copy(); d2.loc[50:, "s_mean"] += 5; b = st.features(d2)
    cols = ["s_mean", "shock20", "d1", "m5", "m20"]; assert np.allclose(a.loc[:49, cols], b.loc[:49, cols])
def test_label_momentum_uses_only_known_labels():
    d = daily(80); a = st.features(d); d2 = d.copy(); d2.loc[60:, "label"] = 1 - d2.loc[60:, "label"]; b = st.features(d2)
    assert np.allclose(a.loc[:59, "mom20"], b.loc[:59, "mom20"])
def test_weekday_dummies_describe_target_day():
    f = st.features(daily(30)); assert ((f[["dow1", "dow2", "dow3", "dow4"]].sum(axis=1)) <= 1).all()
def test_walk_forward_length_and_range():
    f = st.features(daily()); p = st.walk_forward(f, ["s_mean"]); assert len(p) == len(f) - st.START and ((p >= 0) & (p <= 1)).all()
def test_holm_known_values():
    out = st.holm({"a": .01, "b": .02, "c": .03}); assert out == pytest.approx({"a": .03, "b": .04, "c": .04})
def test_holm_never_below_raw():
    p = {f"k{i}": v for i, v in enumerate([.2, .04, .5, .001])}; assert all(st.holm(p)[k] >= p[k] for k in p)
def test_perm_p_small_for_perfect_predictor():
    y = np.random.default_rng(0).integers(0, 2, 400); assert st.shift_perm_p(y, y.astype(float), n=300) < .01
def test_perm_p_not_small_for_noise():
    r = np.random.default_rng(1); y = r.integers(0, 2, 400); assert st.shift_perm_p(y, r.random(400), n=300) > .05
def test_null_se_formula():
    y = np.r_[np.ones(50), np.zeros(50)]; assert st.null_se(y) == pytest.approx(np.sqrt(101 / (12 * 2500)))
def test_planted_signal_is_found_and_null_is_not():
    f = st.features(daily(1500, beta=.6)); y = f.y.values[st.START:]; auc_planted = st.roc_auc_score(y, st.walk_forward(f, ["s_mean"]))
    g = st.features(daily(1500, beta=0, seed=3)); auc_null = st.roc_auc_score(g.y.values[st.START:], st.walk_forward(g, ["s_mean"]))
    assert auc_planted > .58 and abs(auc_null - .5) < .06
def test_power_study_detects_strong_signal():
    pw = st.power_study(st.features(daily(1500)), betas=(0, .8), sims=12); assert pw.detection_rate.iloc[1] > .8 and pw.detection_rate.iloc[0] < .4
