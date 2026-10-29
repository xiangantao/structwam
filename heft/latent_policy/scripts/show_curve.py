#!/usr/bin/env python3
"""Print speed and loss trend from the newest train_*.log in a run directory."""
import json, sys, glob, os
d = sys.argv[1] if len(sys.argv) > 1 else "."
logs = sorted(glob.glob(os.path.join(d, "train_*.log")), key=os.path.getmtime)
rows = []
for l in open(logs[-1], errors="ignore"):
    if l.startswith('{"step"'):
        try: rows.append(json.loads(l))
        except Exception: pass
if not rows: sys.exit("no step lines yet in " + logs[-1])
print(f"{len(rows)} steps logged from {os.path.basename(logs[-1])}")
tot = [r["total"] for r in rows]; el = [r["elapsed_seconds"] for r in rows]
rate = [b - a for a, b in zip(el, el[1:])]
if rate:
    w = rate[-20:]; print(f"speed (last {len(w)} steps): {sum(w)/len(w):.1f} s/step; peak mem {rows[-1].get('peak_allocated_gib', 0):.1f} GiB")
    left = 1000 - rows[-1]["step"]; print(f"eta for 1000 steps: {left * sum(w)/len(w) / 3600:.1f} h left")
n = max(1, len(rows) // 10)
print("total loss, mean per decile:", [round(sum(tot[i:i + n]) / len(tot[i:i + n]), 2) for i in range(0, len(tot), n)])
heads = sorted({k for r in rows for k in r if k.endswith("/total")})
for h in heads:
    v = [r[h] for r in rows if h in r]
    print(f"  {h:22s} first10 {sum(v[:10])/len(v[:10]):7.3f}  last10 {sum(v[-10:])/len(v[-10:]):7.3f}")
