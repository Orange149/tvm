import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

summary_text = r"""
=== Summary (top by total latency) ===
NORMAL     bs=128  ch=256  total=  3.6836ms pack%= 21.2 copy%= 11.7 gops=  6.79
GEMM_ONLY  bs=128  ch=256  total=  3.4950ms pack%= 19.0 copy%= 13.0 gops=  7.07
NORMAL     bs=128  ch=128  total=  2.6662ms pack%= 33.3 copy%= 41.1 gops=  6.13
NORMAL     bs=256  ch=128  total=  2.2657ms pack%= 22.7 copy%= 20.1 gops=  6.47
GEMM_ONLY  bs=256  ch=128  total=  2.1231ms pack%= 27.5 copy%= 17.9 gops=  7.24
GEMM_ONLY  bs=128  ch=128  total=  1.3563ms pack%= 30.4 copy%= 27.9 gops=  7.43
LOAD_INP   bs=128  ch=256  total=  1.1471ms pack%= 67.5 copy%= 31.1 gops=1084.98
ALU_ONLY   bs=128  ch=256  total=  1.1031ms pack%= 64.7 copy%= 31.9 gops=445.81
LOAD_WGT   bs=128  ch=256  total=  1.0569ms pack%= 62.5 copy%= 36.8 gops=2283.13
STORE_OUT  bs=128  ch=256  total=  1.0312ms pack%= 64.1 copy%= 34.0 gops=865.69
"""

# ---------- parse ----------
pat = re.compile(
    r'^(?P<case>\S+)\s+bs=(?P<bs>\d+)\s+ch=(?P<ch>\d+)'
    r'\s+total=\s*(?P<total>[0-9.]+)ms'
    r'\s+pack%=\s*(?P<pack>[0-9.]+)'
    r'\s+copy%=\s*(?P<copy>[0-9.]+)'
    r'\s+gops=\s*(?P<gops>[0-9.]+)\s*$'
)

rows = []
for line in summary_text.splitlines():
    line = line.strip()
    m = pat.match(line)
    if not m:
        continue
    d = m.groupdict()
    rows.append({
        "case": d["case"],
        "bs": int(d["bs"]),
        "ch": int(d["ch"]),
        "total_ms": float(d["total"]),
        "pack_pct": float(d["pack"]),
        "copy_pct": float(d["copy"]),
        "gops": float(d["gops"]),
    })

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No rows parsed. Check the summary text format.")

# label for x-axis
df["label"] = df.apply(lambda r: f'{r["case"]}\nbs={r["bs"]},ch={r["ch"]}', axis=1)

# estimate ms breakdown
df["pack_ms"] = df["total_ms"] * df["pack_pct"] / 100.0
df["copy_ms"] = df["total_ms"] * df["copy_pct"] / 100.0
df["other_ms"] = df["total_ms"] - df["pack_ms"] - df["copy_ms"]

# keep the same order as summary (already sorted by your tool output)
x = np.arange(len(df))

# ---------- fig 1: stacked time breakdown ----------
plt.figure(figsize=(12, 6))
plt.bar(x, df["pack_ms"], label="pack (ms)")
plt.bar(x, df["copy_ms"], bottom=df["pack_ms"], label="copy (ms)")
plt.bar(x, df["other_ms"], bottom=df["pack_ms"] + df["copy_ms"], label="other (ms)")
plt.xticks(x, df["label"], rotation=45, ha="right")
plt.ylabel("Time (ms)")
plt.title("VTA GEMM: time decomposition (estimated from total & percentages)")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- fig 2: pack% / copy% ----------
plt.figure(figsize=(12, 5))
plt.plot(x, df["pack_pct"], marker="o", label="pack%")
plt.plot(x, df["copy_pct"], marker="o", label="copy%")
plt.xticks(x, df["label"], rotation=45, ha="right")
plt.ylabel("Percent of total (%)")
plt.title("VTA GEMM: pack% and copy%")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- fig 3: total vs gops ----------
plt.figure(figsize=(8, 5))
plt.scatter(df["total_ms"], df["gops"])
for _, r in df.iterrows():
    plt.annotate(r["case"], (r["total_ms"], r["gops"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.xlabel("Total latency (ms)")
plt.ylabel("GOPS (as reported)")
plt.title("Total latency vs GOPS")
plt.tight_layout()
plt.show()
