"""Statistical analysis of port incident and offence records.

Runs on the synthetic data from the Port Safety Intelligence Dashboard
(https://github.com/kobbyRollz/port-safety-intelligence-dashboard).

    python analysis.py

Writes figures to figures/ and prints the numbers quoted in README.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10})
BLUE, RED, AMBER, GREY = "#2a78d6", "#d03b3b", "#eda100", "#8a94a6"

inc = pd.read_json(ROOT / "data/demo_incidents.json")
inc["date"] = pd.to_datetime(inc["date"])
inc["harmed"] = inc["kind"].isin(["Injury", "Fatality"])
off = pd.read_csv(ROOT / "data/demo_offences.csv", parse_dates=["date", "date_paid"])

# ---------------------------------------------------------------- 1. control chart
monthly = inc.set_index("date").resample("MS").size()
cbar = monthly.mean()
ucl, lcl = cbar + 3 * np.sqrt(cbar), max(0, cbar - 3 * np.sqrt(cbar))
out = monthly[(monthly > ucl) | (monthly < lcl)]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(monthly.index, monthly.values, marker="o", color=BLUE, lw=1.8)
ax.axhline(cbar, color=GREY, lw=1, label=f"Mean c̄ = {cbar:.1f}")
ax.axhline(ucl, color=RED, ls="--", lw=1, label=f"UCL = {ucl:.1f}")
ax.axhline(lcl, color=RED, ls="--", lw=1, label=f"LCL = {lcl:.1f}")
ax.scatter(out.index, out.values, color=RED, zorder=3, s=60)
ax.set(title="Monthly incidents: c-chart (Poisson control limits)", ylabel="Incidents")
ax.legend(loc="upper right", frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(FIG / "1_control_chart.png"); plt.close(fig)
print(f"[1] c-bar={cbar:.2f}, UCL={ucl:.2f}, LCL={lcl:.2f}, months out of control: "
      f"{[d.strftime('%b %Y') for d in out.index] or 'none'}")

# ---------------------------------------------------------------- 2. rate change 2025 vs 2026
y25 = inc[inc.date.dt.year == 2025]; y26 = inc[inc.date.dt.year == 2026]
m25 = 12; m26 = inc[inc.date.dt.year == 2026].date.dt.month.max()
n25, n26 = len(y25), len(y26)
# conditional test: given n25+n26 events, n26 ~ Binomial(n, m26/(m25+m26)) under equal rates
test = stats.binomtest(n26, n25 + n26, m26 / (m25 + m26))
rr = (n26 / m26) / (n25 / m25)
print(f"[2] 2025: {n25} incidents ({n25/m25:.2f}/month); 2026 to date: {n26} ({n26/m26:.2f}/month); "
      f"rate ratio {rr:.2f}, exact p = {test.pvalue:.3f}")

# ---------------------------------------------------------------- 3. Pareto of mechanisms
mech = inc["mech"].value_counts()
cum = mech.cumsum() / mech.sum() * 100
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(mech.index, mech.values, color=BLUE)
ax.set_ylabel("Incidents"); ax.tick_params(axis="x", rotation=35)
plt.setp(ax.get_xticklabels(), ha="right")
ax2 = ax.twinx(); ax2.plot(mech.index, cum.values, color=RED, marker="o"); ax2.set_ylim(0, 105)
ax2.axhline(80, color=GREY, ls=":", lw=1); ax2.set_ylabel("Cumulative %"); ax2.grid(False)
ax.set_title("What causes incidents? Pareto of incident mechanisms")
fig.tight_layout(); fig.savefig(FIG / "2_pareto_mechanisms.png"); plt.close(fig)
top = cum[cum <= 80].index.tolist() or [mech.index[0]]
print(f"[3] {len(top)} of {len(mech)} mechanisms cause {cum.iloc[len(top)-1]:.0f}% of incidents: {top}")

# ---------------------------------------------------------------- 4. who gets hurt, and where
harm = inc[inc.harmed]
by_loc = inc.groupby("loc").agg(incidents=("id", "size"), harmed=("harmed", "sum")).sort_values("incidents")
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh(by_loc.index, by_loc.incidents, color="#cde2fb", label="All incidents")
ax.barh(by_loc.index, by_loc.harmed, color=BLUE, label="Someone harmed")
ax.set(title="Incident hotspots by location", xlabel="Incidents"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "3_hotspots.png"); plt.close(fig)
roles = (harm["role"].value_counts(normalize=True) * 100).round(0)
top3 = by_loc.incidents.iloc[-3:].sum() / len(inc) * 100
print(f"[4] top location {by_loc.index[-1]} ({by_loc.incidents.iloc[-1]} incidents); top 3 locations hold {top3:.0f}% "
      f"of incidents; people harmed by role: {roles.to_dict()}")

# ---------------------------------------------------------------- 5. Poisson regression + forecast
df = monthly.rename("n").to_frame()
df["t"] = np.arange(len(df))
X = sm.add_constant(df[["t"]])
pois = sm.GLM(df["n"], X, family=sm.families.Poisson()).fit()
disp = pois.pearson_chi2 / pois.df_resid
fut = pd.DataFrame({"t": np.arange(len(df), len(df) + 3)},
                   index=pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=3, freq="MS"))
pred = pois.get_prediction(sm.add_constant(fut[["t"]], has_constant="add")).summary_frame()
mu = pred["mean"].values
lo, hi = stats.poisson.ppf(.05, mu), stats.poisson.ppf(.95, mu)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(df.index, df.n, marker="o", color=BLUE, lw=1.5, label="Observed")
ax.plot(df.index, pois.fittedvalues, color=GREY, lw=1.5, label="Poisson trend")
ax.plot(fut.index, mu, color=AMBER, marker="o", lw=2, label="Forecast")
ax.fill_between(fut.index, lo, hi, color=AMBER, alpha=.2, label="90% prediction interval")
ax.set(title="Monthly incidents: Poisson regression trend and 3-month forecast", ylabel="Incidents")
ax.legend(frameon=False, ncol=4, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "4_forecast.png"); plt.close(fig)
nb = sm.GLM(df["n"], X, family=sm.families.NegativeBinomial(alpha=max(1e-3, (disp - 1) / cbar))).fit()
b = pois.params["t"]
print(f"[5] trend {np.expm1(b)*100:+.1f}% per month (p = {pois.pvalues['t']:.3f}), dispersion = {disp:.2f}; "
      f"negative-binomial trend p = {nb.pvalues['t']:.3f}; forecast {', '.join(f'{d:%b %Y}: {m:.1f} [{l:.0f}-{h:.0f}]' for d, m, l, h in zip(fut.index, mu, lo, hi))}")

# ---------------------------------------------------------------- 6. corrective actions
prev = inc["prev"].value_counts(normalize=True) * 100
print(f"[6] corrective actions: " + ", ".join(f"{k} {v:.0f}%" for k, v in prev.items()))

# ---------------------------------------------------------------- 7. fines
paid = off[off.status == "Paid"]
delay = (paid.date_paid - paid.date).dt.days
status = off.status.value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(9, 3.4))
codes = off.code.value_counts().head(8)
ax.bar(codes.index, codes.values, color=BLUE)
ax.set(title="Most common offence codes", ylabel="Cases")
fig.tight_layout(); fig.savefig(FIG / "5_offences.png"); plt.close(fig)
repeat = off.offender.map(off.offender.value_counts()) > 1
print(f"[7] {len(off)} offences; status: " + ", ".join(f"{k} {v:.0f}%" for k, v in status.items())
      + f"; paid same day {np.mean(delay == 0)*100:.0f}%, median delay {delay.median():.0f} days; "
      f"repeat offenders account for {repeat.mean()*100:.0f}% of cases")
