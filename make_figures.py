import sys, json, sqlite3, ast
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent
B, O, G, INK, MUT = "#2a78d6", "#eb6834", "#9a9a95", "#1a1a19", "#6b6a66"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
  "axes.edgecolor": "#bbb", "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.titleweight": "bold",
  "axes.titlelocation": "left", "axes.titlesize": 13, "figure.dpi": 100, "savefig.dpi": 200, "axes.grid": True,
  "grid.color": "#e8e8e4", "grid.linewidth": .7, "axes.axisbelow": True})
def save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); print(p)

# ---- MarketMind
m = R; img = m/"docs"/"img"
r = pd.read_csv(m/"reports/results.csv"); r = r[r.model != "always_up"].copy()
r["lo"] = r.gain_95ci.apply(lambda s: ast.literal_eval(s)[0] if isinstance(s, str) else np.nan)
r["hi"] = r.gain_95ci.apply(lambda s: ast.literal_eval(s)[1] if isinstance(s, str) else np.nan)
r["lab"] = r.model.str.replace("_logreg", "").str.replace("_", " ") + " (" + r.framing.str.replace("_", "-") + ")"
r = r.iloc[::-1]; fig, ax = plt.subplots(figsize=(7.5, 4.4)); y = np.arange(len(r))
ax.axvline(0, color=INK, lw=1.2); ax.errorbar(r.gain_vs_always_up*100, y, xerr=[(r.gain_vs_always_up-r.lo)*100, (r.hi-r.gain_vs_always_up)*100], fmt="o", color=B, ecolor=B, ms=8, mec="white", mew=1.5, capsize=3)
ax.set_yticks(y); ax.set_yticklabels(r.lab); ax.set_xlabel("Accuracy gain vs always predicting 'up' (percentage points, 95% CI)")
ax.set_title("No model reliably beats the baseline"); ax.grid(axis="y", visible=False); save(fig, img/"gain_vs_baseline.png")

ds = pd.read_csv(m/"reports/daily_sentiment.csv"); ds["q"] = pd.qcut(ds.s_mean, 5, labels=["Q1\nmost negative", "Q2", "Q3", "Q4", "Q5\nleast negative"])
u = ds.groupby("q", observed=True).label.mean()*100
fig, ax = plt.subplots(figsize=(6.8, 4.2)); ax.bar(range(5), u.values, color=B, width=.6)
ax.axhline(ds.label.mean()*100, color=O, lw=2, ls="--", label=f"Overall up-rate {ds.label.mean()*100:.1f}%")
for i, v in enumerate(u.values): ax.text(i, 3, f"{v:.0f}%", ha="center", color="white", fontweight="bold")
ax.legend(frameon=False, loc="upper right")
ax.set_xticks(range(5)); ax.set_xticklabels(u.index); ax.set_ylim(0, 80); ax.set_ylabel("Days Dow closed up (%)"); ax.grid(axis="x", visible=False)
ax.set_title("Up-rate is flat across sentiment quintiles"); save(fig, img/"sentiment_quintiles.png")

