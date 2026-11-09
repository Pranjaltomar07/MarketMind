"""Parse the wide CSV (Top1..Top25 per day) into two tables: days(date,label) and headlines(date,rank,text).
Headlines are stored as Python byte-string literals (b'...'), which are decoded here."""
import ast, sqlite3, pandas as pd
from config import *

def clean_text(x) -> str:
    if not isinstance(x, str): return ""
    x = x.strip()
    if x[:2] in ("b'", 'b"'):
        try:
            v = ast.literal_eval(x); return (v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)).strip()
        except (ValueError, SyntaxError):
            return x[2:-1].strip()
    return x

def build(csv=RAW_CSV):
    df = pd.read_csv(csv); df["Date"] = pd.to_datetime(df.Date, dayfirst=True)
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    tops = [c for c in df.columns if c.startswith("Top")]
    days = df[["Date", "Label"]].rename(columns={"Date": "date", "Label": "label"})
    long = df.melt(id_vars="Date", value_vars=tops, var_name="rank", value_name="raw")
    long["rank"] = long["rank"].str[3:].astype(int); long["text"] = long.raw.map(clean_text)
    long = long[long.text != ""].rename(columns={"Date": "date"})[["date", "rank", "text"]].sort_values(["date", "rank"])
    return days, long

def run(csv=RAW_CSV, db=DB_PATH):
    days, long = build(csv); con = sqlite3.connect(db)
    for d in (days, long): d["date"] = d.date.dt.strftime("%Y-%m-%d")
    days.to_sql("days", con, if_exists="replace", index=False); long.to_sql("headlines", con, if_exists="replace", index=False)
    con.close(); print(f"days={len(days)} headlines={len(long)} range={days.date.min()}..{days.date.max()} up_share={days.label.mean():.3f}")

if __name__ == "__main__":
    run()
