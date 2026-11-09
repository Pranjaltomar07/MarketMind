import sqlite3, json, ast
from pathlib import Path
import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
ROOT = Path(__file__).parent; R = ROOT / "reports"
st.set_page_config("MarketMind", layout="wide"); con = sqlite3.connect(ROOT / "data" / "marketmind.db")
st.title("MarketMind - does world-news sentiment predict the Dow Jones?")
st.caption("Real data: top-25 Reddit r/worldnews headlines per day with Dow Jones up/down labels, 2008-2016 (1,989 days, 49,718 headlines).")
t1, t2, t3, t4 = st.tabs(["Data & sentiment", "Walk-forward results", "What words 'predict'?", "Ingestion"])
d = pd.read_csv(R / "daily_sentiment.csv", parse_dates=["date"])
with t1:
    c = st.columns(3); c[0].metric("Days", f"{len(d):,}"); c[1].metric("Share of up days", f"{d.label.mean():.1%}"); c[2].metric("Mean headline sentiment", f"{d.s_mean.mean():.2f}")
    st.plotly_chart(px.line(d.assign(roll=d.s_mean.rolling(30).mean()), x="date", y="roll", title="30-day average headline sentiment (VADER + finance lexicon)"), width="stretch")
    q = d.assign(quintile=pd.qcut(d.s_mean, 5, labels=["Q1 most negative", "Q2", "Q3", "Q4", "Q5 most positive"])).groupby("quintile", observed=True).label.mean().reset_index()
    f = px.bar(q, x="quintile", y="label", title="Share of up days by same-day sentiment quintile (no monotone pattern)"); f.add_hline(y=d.label.mean(), line_dash="dash"); st.plotly_chart(f, width="stretch")
    st.dataframe(pd.read_sql("select date, rank, text from headlines order by random() limit 8", con), hide_index=True)
with t2:
    r = pd.read_csv(R / "results.csv"); st.dataframe(r, hide_index=True)
    r = r[r.model != "always_up"].copy(); r[["lo", "hi"]] = r.gain_95ci.str.strip("[]").str.split(",", expand=True).astype(float)
    r["label"] = r.framing + " / " + r.model
    f = go.Figure(go.Scatter(x=r.gain_vs_always_up, y=r.label, mode="markers", error_x=dict(type="data", symmetric=False, array=r.hi - r.gain_vs_always_up, arrayminus=r.gain_vs_always_up - r.lo)))
    f.add_vline(x=0, line_dash="dash"); f.update_layout(title="Accuracy gain over 'always predict up' with 95% block-bootstrap CI", xaxis_title="gain in accuracy")
    st.plotly_chart(f, width="stretch")
    st.info("Every interval includes zero or is negative: no model reliably beats always-up. 'same_day' is optimistic (text can post-date the close); 'next_day' is the strict, tradable framing.")
with t3:
    st.dataframe(pd.read_csv(R / "top_words_next_day.csv"), hide_index=True)
    st.warning("These words have no plausible causal link to the Dow (e.g. country names). They are what a flexible model fits to noise, and they are the reason out-of-time testing matters.")
with t4:
    st.json(json.loads((R / "ingest_stats.json").read_text())); st.caption("In-process queue + 4 worker threads on one machine. Not a Kafka benchmark.")
