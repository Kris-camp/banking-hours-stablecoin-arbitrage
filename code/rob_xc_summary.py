"""Assemble the cross-market summary table (tables/rob_xc_summary.tex) from the per-market
outputs rob_xc_{eur,zar,pln}.json, emp_mxn.json and rob_mexico_utc.json. The payment-rail column
(whether a USD 50,000 transfer can settle at 02:00 local time) is entered here from the institutional
sources cited in the paper. Same specification in every row: five-minute averages, bands by month x local
banking state, month-by-state effects, both sides of the band pooled, dependent variable -s(db - beta*OFI);
business hours 09:00-17:00 local time."""
import json, math, os
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "")
L = lambda f: json.load(open(T + f))

eur, zar, pln, mx, em = L("rob_xc_eur.json"), L("rob_xc_zar.json"), L("rob_xc_pln.json"), L("rob_mexico_utc.json"), L("emp_mxn.json")
# mean USDT volume per minute in each estimation sample, thousands
V = lambda v: f"{v / 1000:.1f}"
tr_full = eur["diff_vs_TRY"]["TRY"]["pooled"]
tr_win, mxn = mx["local_baseline"]["TRY"], mx["local_baseline"]["MXN"]

def row(name, rail, window, vol, closed, incr, logr):
    f = lambda v: f"${v[0]:.4f}$ $({v[1]:.4f})$".replace("$-", "$-")
    ratio = (closed[0] + incr[0]) / closed[0]
    return f"{name} & {rail} & {window} & {vol} & {f(closed)} & {f(incr)} & {ratio:.1f} & ${logr[0]:.2f}$ $({logr[1]:.2f})$ \\\\"

rows = [
    row("Turkey", "Limited", "1/2024--8/2026", V(eur["diff_vs_TRY"]["TRY"]["vol_per_min"]), tr_full["closed"], tr_full["incr"], tr_full["logratio"]),
    row("Turkey", "Limited", "11/2024--8/2026", V(em["TRY"]["vol_per_min"]), tr_win["closed"], tr_win["incr"], tr_win["log_ratio"]),
    row("Mexico", "Limited", "11/2024--8/2026", V(em["MXN"]["vol_per_min"]), mxn["closed"], mxn["incr"], mxn["log_ratio"]),
    row("Euro area", "Yes", "1/2024--8/2026", V(eur["premium"]["vol_per_min"]), [eur["kappa_closed"], eur["kappa_closed_se"]], [eur["increment"], eur["increment_se"]], [eur["logratio"], eur["logratio_se"]]),
    row("South Africa", "Yes", "1/2024--8/2026", V(zar["vol_per_min"]), [zar["kappa_closed"], zar["kappa_closed_se"]], [zar["increment"], zar["increment_se"]], [zar["logratio"], zar["logratio_se"]]),
    row("Poland", "Limited", "1/2024--5/2026", V(pln["premium"]["vol_per_min_in_sample"]), [pln["kappa_closed"], pln["kappa_closed_se"]], [pln["increment"], pln["increment_se"]], [pln["logratio"], pln["logratio_se"]]),
]
tex = r"""\begin{tabular}{lllcccccc}
\toprule
 & USD 50{,}000 & & Volume per & \multicolumn{2}{c}{Speed outside the band} & Open/ & \\
\cmidrule(lr){5-6}
Market & at 02:00 & Sample & minute & Banks closed & $\times$ banks open & closed & Log ratio \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
tex = tex.replace(r"\begin{tabular}{lllcccccc}", r"\begin{tabular}{lllccccc}")
open(T + "rob_xc_summary.tex", "w").write(tex)
print(tex)
