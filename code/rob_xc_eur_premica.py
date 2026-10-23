"""Euro-area row of Table rob_xc before Binance removed USDT spot pairs for users in the European
Economic Area (31 March 2025): same construction as code/rob_xc_eur.py, January 2024 - March 2025.

Inputs: as rob_xc_eur.py (imported; its own estimation does not run on import).
Output: tables/rob_xc_eur_premica.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rob_xc_eur as E

a = E.build("EUR", "2024-01-01", "2025-03-31")
out, _ = E.estimate(a)
out["window"] = ["2024-01-01", "2025-03-31"]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "rob_xc_eur_premica.json"), "w"), indent=1)
p = out["pooled"]
print(out["window"], out["n"], out["months"][0], out["months"][-1], "vol/min", round(out["vol_per_min"], 1))
print("closed", p["closed"], "incr", p["incr"], "ratio", round(p["ratio"], 2), "logratio", p["logratio"])
