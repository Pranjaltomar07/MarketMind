"""Figures for the stronger null-result tests. Run after src/signal_tests.py."""
import ast, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent; IMG = R / "docs" / "img"; REP = R / "reports"
AMBER, GREEN, INK, MUT, GREY = "#e8a317", "#2f9e73", "#1a1a19", "#6b6a66", "#9a9a95"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bbb", "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 13,
                     "savefig.dpi": 200, "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": .7, "axes.axisbelow": True})
def save(fig, name): fig.savefig(IMG / name, bbox_inches="tight", facecolor="white"); plt.close(fig); print(name)

IMG.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(REP / "signal_tests.csv"); ci = t.auc_95ci.apply(ast.literal_eval); t = t.iloc[::-1].reset_index(drop=True); ci = ci.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.4, 4.3)); y = np.arange(len(t))
ax.axvline(.5, color=GREY, ls="--", lw=1.5); ax.errorbar(t.auc, y, xerr=[t.auc - [c[0] for c in ci], [c[1] for c in ci] - t.auc], fmt="o", color=AMBER, ecolor=MUT, capsize=3, ms=8, mec="white")
ax.set_yticks(y); ax.set_yticklabels(t.feature_set.str.replace("_", " ")); ax.set_xlabel("AUC on 988 later days (0.5 = coin flip), 95% block-bootstrap interval")
ax.set_title("Seven kinds of feature, none better than a coin flip"); ax.grid(axis="y", visible=False); save(fig, "signal_auc.png")

p = pd.read_csv(REP / "signal_power.csv"); s = json.loads((REP / "signal_power_summary.json").read_text())
fig, ax = plt.subplots(figsize=(6.6, 4.2)); ax.plot(p.mean_auc, p.detection_rate * 100, "-o", color=GREEN, lw=2.2, ms=8, mec="white")
ax.axhline(80, color=GREY, ls="--", lw=1.2); ax.text(p.mean_auc.min(), 82, "80% power", color=MUT, fontsize=10)
ax.axhline(5, color=GREY, ls=":", lw=1.2); ax.text(p.mean_auc.max(), 7, "false-alarm rate when nothing is planted", color=MUT, fontsize=10, ha="right")
ax.set_xlabel("AUC the planted signal would give"); ax.set_ylabel("how often the test detects it, %"); ax.set_title("What this test could and could not see")
save(fig, "signal_power.png")

d = pd.read_csv(REP / "daily_sentiment.csv", parse_dates=["date"]).set_index("date")
fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.plot(d.s_mean.rolling(60).mean(), color=AMBER, lw=2); ax.plot(d.s_mean, color=GREY, lw=.5, alpha=.5)
ax.set_ylabel("mean headline sentiment"); ax.set_title("Daily sentiment (grey) and 60-day average"); save(fig, "sentiment_over_time.png")
