"""
Read the surface as an options desk would, then say what that means for US health.

Everything here is historical frequency on self-reported data. The analogies are a way to organize the
results: nothing trades, there's no arbitrage argument, and none of this shows what causes obesity.

Outputs:
  - outputs/lens_term_structure.csv: forwards, crossing odds and sigma, trend vs flat
  - outputs/lens_greeks.csv: delta, gamma, theta and vega at the national spot
  - outputs/lens_variance.csv: survey noise vs real change in the state-level variance
  - outputs/lens_states.csv: each state's level, moneyness and odds
  - outputs/lens_state_counts.csv: expected number of states over each level
  - outputs/lens_cohorts.csv: generations followed through the age bands
  - outputs/lens_cohort_forward.csv: what aging alone implies for the national rate
  - outputs/lens_behaviors.csv: diet and activity, nationally and against obesity
  - outputs/lens_value.csv: what a point of obesity means in adults and medical costs
  - outputs/lens_forward_curve.png, outputs/lens_generations.png
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "outputs"
spec = importlib.util.spec_from_file_location("surface", PROJECT / "scripts" / "02_build_surface.py")
surface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface)
logit, expit, EXCLUDE = surface.logit, surface.expit, surface.EXCLUDE

TS = surface.TS
AGES = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
COST_PER_ADULT = 1861  # CDC, annual excess medical cost, 2019 dollars
TOTAL_COST_B = 173  # CDC, annual medical cost of adult obesity, billions, 2019 dollars


def crossing(m, k, s):
    return norm.cdf((m - logit(k)) / s)


def term_structure(curves, S0):
    rows = []
    for T, c in curves.items():
        m = c["a"] + c["b"] * S0
        rows.append({"T": T, "forward_pct": expit(m) * 100, "sigma_annualized": c["s"] / np.sqrt(T), "total_sd_logit": c["s"], "n": c["n"],
                     **{f"p{int(k * 100)}_trend": crossing(m, k, c["s"]) for k in (0.35, 0.38, 0.40)},
                     **{f"p{int(k * 100)}_flat": crossing(S0, k, c["s"]) for k in (0.35, 0.38, 0.40)}})
    return pd.DataFrame(rows)


def greeks(curves, S0, p0):
    """Per point of prevalence today, so delta reads as: one point lower now changes the odds by this much."""
    rows = []
    for K in (0.35, 0.38, 0.40):
        for T, c in curves.items():
            m = c["a"] + c["b"] * S0
            base = crossing(m, K, c["s"])
            bump = lambda p: crossing(c["a"] + c["b"] * logit(p), K, c["s"])
            h = 0.005
            delta = (bump(p0 + h) - bump(p0 - h)) / (2 * h) / 100
            gamma = (bump(p0 + h) - 2 * base + bump(p0 - h)) / h ** 2 / 100 ** 2
            vega = crossing(m, K, c["s"] * 1.1) - base
            rows.append({"K": K, "T": T, "p": base, "delta_per_point": delta, "gamma_per_point2": gamma, "vega_sigma_up_10pct": vega})
    out = pd.DataFrame(rows).sort_values(["K", "T"])
    # Theta: the odds gained per added year of waiting, between listed horizons.
    out["theta_per_year"] = out.groupby("K").p.diff() / out.groupby("K")["T"].diff()
    return out


def variance_split(total, curves):
    """Total variance against horizon. The intercept is noise that doesn't build with time, the slope is real yearly variance."""
    states = total[~total.location.isin(EXCLUDE)].copy()
    states["se_logit"] = (logit(states.high / 100) - logit(states.low / 100)) / (2 * 1.96)
    survey_var = float((states.se_logit ** 2).mean())
    t = np.array(list(curves))
    var = np.array([curves[x]["s"] ** 2 for x in t])
    slope, intercept = np.polyfit(t, var, 1)
    return pd.DataFrame([{"intercept_var": intercept, "slope_var_per_year": slope, "sd_noise": np.sqrt(max(intercept, 0)), "sd_real_per_year": np.sqrt(max(slope, 0)),
                          "sd_survey_error_one_year": np.sqrt(survey_var), "sd_survey_error_both_ends": np.sqrt(2 * survey_var),
                          "share_noise_at_1y": min(1, intercept / var[0]), "share_noise_at_5y": intercept / var[list(t).index(5)], "share_noise_at_10y": intercept / var[-1]}])


def states_table(total, curves):
    wide = total[~total.location.isin(EXCLUDE)].pivot(index="location", columns="year", values="percent")
    level = wide.ffill(axis=1).iloc[:, -1]
    last_year = wide.apply(lambda r: r.last_valid_index(), axis=1)
    rows = []
    c = curves[5]
    for state, value in level.items():
        s0 = logit(value / 100)
        m = c["a"] + c["b"] * s0
        row = {"state": state, "latest": value, "latest_year": int(last_year[state]), "points_to_35": 35 - value, "points_to_40": 40 - value,
               "forward_5y": expit(m) * 100, "p35_5y": crossing(m, 0.35, c["s"]), "p40_5y": crossing(m, 0.40, c["s"]), "p40_5y_flat": crossing(s0, 0.40, c["s"])}
        h = 0.005
        p = value / 100
        row["delta40_per_point"] = (crossing(c["a"] + c["b"] * logit(p + h), 0.40, c["s"]) - crossing(c["a"] + c["b"] * logit(p - h), 0.40, c["s"])) / (2 * h) / 100
        rows.append(row)
    return pd.DataFrame(rows).sort_values("latest", ascending=False)


def state_counts(total, curves):
    wide = total[~total.location.isin(EXCLUDE)].pivot(index="location", columns="year", values="percent")
    level = wide.ffill(axis=1).iloc[:, -1] / 100
    rows = [{"T": 0, "scenario": "now", **{f"over_{int(k * 100)}": float((level >= k).sum()) for k in (0.30, 0.35, 0.40)}}]
    for T, c in curves.items():
        for scenario in ("trend", "flat"):
            m = c["a"] + c["b"] * logit(level) if scenario == "trend" else logit(level)
            rows.append({"T": T, "scenario": scenario, **{f"over_{int(k * 100)}": float(crossing(m, k, c["s"]).sum()) for k in (0.30, 0.35, 0.40)}})
    return pd.DataFrame(rows)


def age_panel(latest):
    states = latest[~latest.location.isin(EXCLUDE) & latest.group.isin(AGES)]
    cube = states.pivot_table(index="location", columns=["group", "year"], values="percent").dropna()
    adults = pd.read_csv(RAW / "census_adults_by_age.csv").pivot(index="state", columns="group", values="adults").reindex(cube.index)
    means = pd.DataFrame({age: (cube[age].multiply(adults[age], axis=0)).sum() / adults[age].sum() for age in AGES})
    return means, adults


def cohorts(means):
    rows = []
    for start in (2011, 2015):
        end = start + 10
        for young, old in zip(AGES[:-1], AGES[1:]):
            rows.append({"start": start, "end": end, "from_age": young, "to_age": old, "rate_start": means.loc[start, young],
                         "rate_end": means.loc[end, old], "gain": means.loc[end, old] - means.loc[start, young]})
    same_age = [{"age": a, "rate_2011": means.loc[2011, a], "rate_2021": means.loc[2021, a], "rate_2025": means.loc[2025, a],
                 "change_2011_2025": means.loc[2025, a] - means.loc[2011, a]} for a in AGES]
    return pd.DataFrame(rows), pd.DataFrame(same_age)


def cohort_forward(means, adults, gains):
    """Roll each age band up ten years, keeping today's age mix. The 18-24 band is held where it is, since those people aren't surveyed yet."""
    weight = adults.sum() / adults.sum().sum()
    now = means.loc[2025, AGES]
    gain = gains[gains.start == 2015].set_index("to_age").gain
    no_gain, same_gain = {"18-24": now["18-24"]}, {"18-24": now["18-24"]}
    for young, old in zip(AGES[:-1], AGES[1:]):
        no_gain[old] = now[young]
        same_gain[old] = now[young] + gain[old]
    frame = lambda scen, values: {"scenario": scen, "national_pct": sum(values[a] * weight[a] for a in AGES), **values}
    out = pd.DataFrame([frame("today", now.to_dict()), frame("2035, no further gain within a generation", no_gain),
                        frame("2035, same gain as 2015-2025", same_gain)])
    today, floor, full = out.national_pct
    # The share of last decade's gain that would leave the national rate where it is.
    out["share_of_gain_that_keeps_rate_flat"] = (today - floor) / (full - floor)
    return out


def behaviors(total):
    b = pd.read_csv(RAW / "cdc_brfss_behaviors.csv")
    wide = b[~b.location.isin(EXCLUDE)].pivot_table(index=["location", "year"], columns="measure", values="percent").reset_index()
    obesity = total[~total.location.isin(EXCLUDE)][["location", "year", "percent"]].rename(columns={"percent": "obesity"})
    d = wide.merge(obesity, on=["location", "year"])
    rows = []
    for measure in ("low_vegetables", "low_fruit", "inactive"):
        frame = d.dropna(subset=[measure, "obesity"])
        balanced = frame.groupby("location").filter(lambda g: g.year.nunique() == frame.year.nunique())
        means = balanced.groupby("year")[measure].mean()
        first, last = frame.year.min(), frame.year.max()
        cross = frame[frame.year == frame[frame[measure].notna()].year.max()]
        fe = smf.ols(f"obesity ~ {measure} + C(location) + C(year)", frame).fit(cov_type="cluster", cov_kwds={"groups": frame.location.astype("category").cat.codes})
        a, z = frame[frame.year == first].set_index("location"), frame[frame.year == last].set_index("location")
        rows.append({"measure": measure, "first_year": first, "last_year": last, "waves": frame.year.nunique(), "states_balanced": balanced.location.nunique(),
                     "mean_first": means.loc[first], "mean_2021": means.get(2021, np.nan), "mean_last": means.loc[last],
                     "cross_state_r_latest": cross[measure].corr(cross.obesity), "fe_coef": fe.params[measure], "fe_t": fe.tvalues[measure], "fe_n": int(fe.nobs),
                     "change_r": (z[measure] - a[measure]).corr(z.obesity - a.obesity), "change_states": int((z[measure] - a[measure]).notna().sum())})
    both = pd.read_csv(RAW / "cdc_brfss_behaviors.csv")
    over = both[(both.measure == "overweight") & ~both.location.isin(EXCLUDE)].pivot(index="location", columns="year", values="percent")
    keep = over.dropna().index
    ob = total[~total.location.isin(EXCLUDE)].pivot(index="location", columns="year", values="percent")
    keep = keep.intersection(ob.dropna().index)
    rows.append({"measure": "overweight_or_obese", "first_year": 2011, "last_year": 2025, "states_balanced": len(keep), "mean_first": (over.loc[keep, 2011] + ob.loc[keep, 2011]).mean(),
                 "mean_2021": (over.loc[keep, 2021] + ob.loc[keep, 2021]).mean(), "mean_last": (over.loc[keep, 2025] + ob.loc[keep, 2025]).mean()})
    return pd.DataFrame(rows)


def value_table(adults_total, p0, curves, S0):
    point = adults_total / 100
    rows = [{"item": "US civilian adults, 2024", "value": adults_total}, {"item": "adults with obesity at the spot", "value": adults_total * p0},
            {"item": "adults per point of obesity", "value": point}, {"item": "annual excess medical cost per point, $", "value": point * COST_PER_ADULT}]
    for T in (5, 10):
        c = curves[T]
        gap = (expit(c["a"] + c["b"] * S0) - p0) * 100
        rows += [{"item": f"forward minus spot at {T}y, points", "value": gap}, {"item": f"adults between trend and flat at {T}y, same adult population", "value": gap * point},
                 {"item": f"annual excess medical cost between trend and flat at {T}y, $", "value": gap * point * COST_PER_ADULT}]
    return pd.DataFrame(rows)


def figures(ts, spot, means):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0] + list(ts["T"]), [spot] + list(ts.forward_pct), color="#b32a18", lw=2, marker="o", ms=6, label="Forward curve, if the old trend holds")
    ax.plot([0, ts["T"].max()], [spot, spot], color="#0f6cbd", lw=2, ls="--", label="Flat, if the plateau since 2021 holds")
    for t, f in zip(ts["T"], ts.forward_pct):
        if t in (1, 5, 10):
            ax.annotate(f"{f:.1f}%", (t, f), textcoords="offset points", xytext=(0, 9), ha="center")
    ax.set(xlabel="Years ahead", ylabel="Self-reported adult obesity (%)", ylim=(31, 40), title="The curve is in contango, and spot has stopped rolling up")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "lens_forward_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {2011: "#9a9484", 2025: "#b32a18"}
    x = range(len(AGES))
    for year, color in colors.items():
        ax.plot(x, means.loc[year, AGES], color=color, lw=2, marker="o", ms=6, label=str(year))
    ax.set_xticks(list(x), AGES)
    ax.set(xlabel="Age", ylabel="Self-reported obesity (%)", title="Every age group reports more obesity than in 2011")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "lens_generations.png", dpi=160)
    plt.close(fig)


def main():
    latest = pd.read_csv(RAW / "cdc_brfss_latest.csv")
    total = latest[latest.group == "Total"]
    spot = surface.adjusted_spot(total)
    p0 = spot["adjusted"] / 100
    S0 = float(logit(p0))
    curves = surface.fit_curves(total)

    ts = term_structure(curves, S0)
    gr = greeks(curves, S0, p0)
    var = variance_split(total, curves)
    st = states_table(total, curves)
    counts = state_counts(total, curves)
    means, adults = age_panel(latest)
    gains, same_age = cohorts(means)
    fwd = cohort_forward(means, adults, gains)
    beh = behaviors(total)
    val = value_table(float(pd.read_csv(RAW / "census_adults_by_age.csv").adults.sum()), p0, curves, S0)

    for name, frame in (("term_structure", ts), ("greeks", gr), ("variance", var), ("states", st), ("state_counts", counts), ("cohort_forward", fwd), ("behaviors", beh), ("value", val)):
        frame.round(4).to_csv(OUT / f"lens_{name}.csv", index=False)
    pd.concat([gains.assign(view="generation"), same_age.assign(view="same_age")]).round(3).to_csv(OUT / "lens_cohorts.csv", index=False)
    figures(ts, spot["adjusted"], means)

    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("Spot", spot)
    print(ts.round(3).to_string(index=False))
    print(gr[gr["T"].isin([1, 5, 10])].round(3).to_string(index=False))
    print(var.round(4).T.to_string())
    print(st.head(12).round(2).to_string(index=False))
    print(st.sort_values("delta40_per_point", ascending=False).head(8)[["state", "latest", "p40_5y", "delta40_per_point"]].round(3).to_string(index=False))
    print(counts.round(1).to_string(index=False))
    print(means.round(1).loc[[2011, 2015, 2021, 2025]].to_string())
    print(gains.round(1).to_string(index=False))
    print(fwd.round(1).to_string(index=False))
    print(beh.round(2).T.to_string())
    print(val.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
