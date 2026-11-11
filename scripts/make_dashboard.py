"""Build reports/dashboard.html: one self-contained page from the files in reports/ (no server, no internet).
Run:  python scripts/make_dashboard.py   then open reports/dashboard.html"""
import ast, base64, html, json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent.parent; REP = R / "reports"; IMG = R / "docs" / "img"
res = pd.read_csv(REP / "results.csv"); sig = pd.read_csv(REP / "signal_tests.csv"); pw = pd.read_csv(REP / "signal_power.csv")
ps = json.loads((REP / "signal_power_summary.json").read_text()); sub = pd.read_csv(REP / "signal_subperiods.csv")
st = json.loads((REP / "ingest_stats.json").read_text()); daily = pd.read_csv(REP / "daily_sentiment.csv", parse_dates=["date"])
AMB, GRN = "#e8a317", "#2f9e73"


def b64(n): p = IMG / n; return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""
def table(df):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>' + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False)) + "</tbody></table></div>"


def line_svg(w=760, h=240):
    m = daily.set_index("date").s_mean.resample("ME").mean(); lo, hi = m.min(), m.max()
    X = lambda i: 46 + i * (w - 66) / (len(m) - 1); Y = lambda v: h - 30 - (h - 50) * (v - lo) / (hi - lo)
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img"><polyline fill="none" stroke="{AMB}" stroke-width="2.2" points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(m.values))}"/>']
    for i in range(0, len(m), 12): s.append(f'<text x="{X(i):.1f}" y="{h-10}" text-anchor="middle" font-size="11">{m.index[i].year}</text>')
    return "".join(s) + f'<text x="4" y="14" font-size="11">monthly mean sentiment, {lo:.2f} to {hi:.2f}</text></svg>'


nd = res[(res.framing == "next_day") & (res.model == "vader_logreg")].iloc[0]; lo, hi = ast.literal_eval(nd.gain_95ci)
best = sig.sort_values("perm_p").iloc[0]
rtab = res.assign(gain_95ci=res.gain_95ci.fillna("")).drop(columns=["n_test_days"]).round(3)
stab = sig[["feature_set", "auc", "auc_95ci", "gain_vs_always_up", "perm_p", "holm_p"]]
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MarketMind dashboard</title>
<style>:root{{--bg:#f6f5f1;--card:#fff;--ink:#1a1a19;--mut:#6b6a66;--line:#e3e2dc;--acc:{GRN}}}@media(prefers-color-scheme:dark){{:root{{--bg:#14171a;--card:#1c2024;--ink:#ececea;--mut:#a4a4a0;--line:#2c3238}}}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink)}}header,nav,main,footer{{max-width:980px;margin:auto;padding:0 20px}}header{{padding-top:22px}}h1{{margin:0;font-size:26px}}header p{{color:var(--mut);margin:4px 0}}
nav{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}}nav button{{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:7px 13px;border-radius:20px;cursor:pointer;font:inherit}}nav button[aria-selected=true]{{background:var(--acc);color:#fff;border-color:var(--acc)}}
section{{display:none;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:10px}}section.on{{display:block}}.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:12px}}.kpi{{border:1px solid var(--line);border-radius:10px;padding:10px 12px}}.kpi b{{display:block;font-size:24px;color:var(--acc)}}.kpi span{{color:var(--mut);font-size:12px}}
.chart{{width:100%;height:auto}}.chart text{{fill:var(--ink)}}img{{max-width:100%;border-radius:8px;background:#fff}}.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}.scroll{{overflow:auto;max-height:380px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}.note{{color:var(--mut)}}footer{{color:var(--mut);font-size:12px;padding:12px 20px 30px}}</style></head><body>
<header><h1>MarketMind: does news sentiment predict the Dow?</h1><p>Real data: top-25 r/worldnews headlines per day with Dow up/down labels, 2008 to 2016. Answer so far: no, and here is how sure we are.</p></header>
<nav role="tablist">{"".join(f'<button role="tab" aria-selected="{str(i==0).lower()}" data-t="t{i}">{n}</button>' for i, n in enumerate(["Overview", "Seven feature sets", "Could it see a signal?", "Sentiment over time"]))}</nav><main>
<section id="t0" class="on"><div class="kpis"><div class="kpi"><b>{st['headlines']:,}</b><span>headlines scored ({st['headlines_per_sec']:,} per second)</span></div><div class="kpi"><b>{len(daily):,}</b><span>trading days</span></div><div class="kpi"><b>{nd.gain_vs_always_up*100:+.1f} pts</b><span>best strict model vs always-up (95% CI {lo*100:+.1f} to {hi*100:+.1f})</span></div><div class="kpi"><b>{best.holm_p:.2f}</b><span>smallest corrected p-value across seven feature sets</span></div></div>
<div class="two"><div><h3>Gain over "always up"</h3><img src="{b64('gain_vs_baseline.png')}" alt="gain over always-up"></div><div><h3>Up-rate by sentiment quintile</h3><img src="{b64('sentiment_quintiles.png')}" alt="quintiles"></div></div><h3>Original comparison</h3>{table(rtab)}</section>
<section id="t1"><h3>Seven kinds of feature, walk-forward, 988 later days</h3><img src="{b64('signal_auc.png')}" alt="AUC by feature set"><p class="note">Permutation p-values keep each series' autocorrelation; Holm correction adjusts for seven looks. Nothing survives it.</p>{table(stab)}</section>
<section id="t2"><h3>Could this test have seen a real signal?</h3><div class="two"><div><img src="{b64('signal_power.png')}" alt="power curve"></div><div><p>With {int(ps['n_test_days'])} test days the AUC has a standard error of about {ps['null_se_auc']:.3f} when there is no signal. A signal must give AUC of about <b>{ps['auc_detectable_5pct']:.3f}</b> to be noticed, and about <b>{ps['auc_for_80pct_power']:.3f}</b> to be noticed 80% of the time.</p><p>We planted signals of known strength in the real sentiment series and ran the same code. With nothing planted it raised a false alarm 5% of the time, as it should. A planted signal worth AUC {pw.mean_auc.iloc[2]:.2f} was found {pw.detection_rate.iloc[2]*100:.0f}% of the time.</p><p class="note">So effects much larger than this are ruled out; very small ones are not.</p></div></div>{table(pw)}</section>
<section id="t3"><h3>Sentiment drifts, and varies little day to day</h3>{line_svg()}<p class="note">That is why the tests also try a "shock" feature (today against its own recent average) instead of only the raw level.</p>{table(sub)}</section>
</main><footer>Reports in reports/ come from running this repository. Not investment advice.</footer>
<script>const tabs=[...document.querySelectorAll('nav button')];tabs.forEach(b=>b.onclick=()=>{{tabs.forEach(x=>x.setAttribute('aria-selected',x===b));document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t))}});</script></body></html>"""
(REP / "dashboard.html").write_text(page, encoding="utf-8"); print("wrote", REP / "dashboard.html")
