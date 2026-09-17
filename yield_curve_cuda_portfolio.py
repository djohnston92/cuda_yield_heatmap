import os
import time
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import cupy as cp
import pandas as pd
import duckdb
from sqlalchemy import create_engine
import psycopg2
from sqlalchemy import create_engine, text
##############################################################################################################################
SHORT_TERM_CUTOFF = 120  # months; only used as a last-resort fallback, see PORTFOLIO_DEFAULTS

MAX_MATRIX_CELLS_GPU = 40_000_000  # cap on (batch_size * (term+1)) per chunk -- sized for the 1080's 8GB VRAM, NOT system RAM like the CPU version's cap

APR_AXIS_RANGE = (0.0, 20.0)  # percent; fixed so plots are comparable across datasets

PD_AXIS_RANGE = (0.0, 25.0)   # percent; fixed so plots are comparable across datasets
YIELD_AXIS_RANGE = (0.0, 6.0)  # percent; fixed color scale so plots are comparable across datasets
control_object_recordid='d10af4c9-dbb6-49c4-9cf3-2d7219972cbe'



lei ='549300XY701IELCE5Q08'
state ='all'
##############################################################################################################################
duck_con = duckdb.connect("/mnt/34bc75ca-9453-4032-88f3-699d8994bd14/DuckDb/localized.duckdb")
m910 = create_engine(f"postgresql+psycopg2://{os.getenv('PG_SUPERUSER_USER')}:{os.getenv('PG_SUPERUSER_PASSWORD')}@{os.getenv('PG_SUPERUSER_HOST')}:{os.getenv('PG_SUPERUSER_PORT')}/{os.getenv('PG_SUPERUSER_DB')}")
save_path_visual = "/home/dad/Documents/CUDA Projects/"
save_name_visual = f"{lei}{state}_gpu.png"
full_path = save_path_visual + save_name_visual
##############################################################################################################################
vintage_examined = f"""
    select
        record_id
        ,activity_year
        ,lei
        ,census_tract
        ,county_fips
        ,state_code
        ,loan_amount
        ,asset_value
        ,ltv
        ,interest_rate_num
        ,loan_term
        ,joint_pd_base
        ,'mortgage' as portfolio_type
        ,total_loan_costs
    from localized.hmda.hdma_merton_vw
    where activity_year=2025
     --   and        state_code='{state}'
       and  lei='{lei}'

"""


##############################################################################################################################
curve_control = f"""
select * from python.yield_heatmap_loan_type_presets
where record_id='{control_object_recordid}'
"""
control_df = pd.read_sql(curve_control,m910)
PORTFOLIO_DEFAULTS = {}
for _, row in control_df.iterrows():
    PORTFOLIO_DEFAULTS[row["loan_type"]] = dict(
        apr_default=row["default_apr"],
        pd_default=row["default_pd"],
        lgd_default=row["lgd"],
        principal_default=row["principal_max"],   # pick whichever makes sense
        term_default=row["term_options"],
        fee_default=row["default_fee"],
        curve_type=row["curve_type"],
        alias=row["alias"],
    )
print(PORTFOLIO_DEFAULTS)
_PORTFOLIO_ALIASES = dict(zip(control_df["loan_type"], control_df["alias"]))
##############################################################################################################################
def default_profile(term, curve_type):
    """Same two shapes as the CPU version's default_profile, but returns a GPU
    (CuPy) array in float32 -- Pascal's FP64 throughput is ~1/32 of float32,
    see reference_cuda_venv_gtx1080 memory."""
    prof = cp.zeros(term + 1, dtype=cp.float32)
    if curve_type == "front_loaded":
        first_year = min(12, term)
        prof[1:first_year + 1] = 1.0 / first_year
    else:  # "stair_step"
        prof[1:37] = 0.60 / 36
        prof[37:85] = 0.25 / 48
        prof[85:term + 1] = 0.15 / max(term - 84, 1)
    prof /= prof.sum()
    return prof
##############################################################################################################################
def _resolve_curve_types(term_months, curve_type_raw):
    """Turn whatever's in curve_type_col (or None) into a clean array of
    'front_loaded' / 'stair_step' strings, one per loan. Stays entirely on the
    CPU -- CuPy has no string dtype, and this is a one-shot vectorized pandas/
    numpy op anyway, not a per-loan loop, so there's nothing to gain by moving
    it to the GPU."""
    default = np.where(term_months < SHORT_TERM_CUTOFF, "front_loaded", "stair_step")

    if curve_type_raw is None:
        return default

    s = pd.Series(curve_type_raw).astype(str).str.strip().str.lower()
    is_front = s.str.contains("front", na=False).to_numpy()
    is_stair = (s.str.contains("stair", na=False) | s.str.contains("step", na=False)).to_numpy()

    resolved = default.copy()
    resolved[is_front] = "front_loaded"
    resolved[is_stair] = "stair_step"
    return resolved
##############################################################################################################################
def _yield_and_gross(principal, fee, term_months, pd_arr, apr, lgd_arr, curve_types):
    """GPU port of the CPU version's core engine -- identical math, same
    batching shape (loop over unique (term, curve_type) groups, full matrix
    ops within each group). The group-by itself stays on the host (pandas),
    same reason as _resolve_curve_types -- curve_types is a string array with
    no CuPy equivalent. Each group's numeric slice is uploaded to the GPU as
    float32 for the actual matrix math, then pulled back for the next group;
    no per-loan Python loop and no per-loan kernel launches, which is the
    difference between this and the old depreciated/ CUDA attempts."""
    n_loans = len(principal)
    yield_percent = np.zeros(n_loans, dtype=np.float32)
    gross_yield_percent = np.zeros(n_loans, dtype=np.float32)

    df_group = pd.DataFrame({"term": term_months, "curve_type": curve_types})
    for (term, curve_type), group in df_group.groupby(["term", "curve_type"]):
        idx_all = group.index.to_numpy()
        n = int(term)
        batch_size = max(1, MAX_MATRIX_CELLS_GPU // (n + 1))

        prof = default_profile(n, curve_type)  # GPU array, (n+1,)

        for start in range(0, idx_all.size, batch_size):
            idx = idx_all[start:start + batch_size]

            apr_g = cp.asarray(apr[idx], dtype=cp.float32)
            pd_g = cp.asarray(pd_arr[idx], dtype=cp.float32)
            principal_g = cp.asarray(principal[idx], dtype=cp.float32)
            fee_g = cp.asarray(fee[idx], dtype=cp.float32)
            lgd_g = cp.asarray(lgd_arr[idx], dtype=cp.float32)

            r_m = apr_g / 12
            t = cp.arange(n + 1, dtype=cp.float32)

            # r_m == 0 is a real case (0% promotional loans exist), and it has
            # a perfectly well-defined amortization schedule -- equal principal
            # payments, zero interest -- but the general formula below hits a
            # literal 0/0 there. Compute it on a safe placeholder rate, then
            # overwrite those specific rows with the correct zero-rate math.
            is_zero_rate = (r_m == 0)
            safe_r_m = cp.where(is_zero_rate, cp.float32(1.0), r_m)  # placeholder, overwritten below

            payment = (safe_r_m * (1 + safe_r_m) ** n) / ((1 + safe_r_m) ** n - 1)
            payment = cp.where(is_zero_rate, cp.float32(1.0 / n), payment)

            growth = (1 + safe_r_m)[:, None] ** t[None, :]
            B = growth - payment[:, None] * (growth - 1) / safe_r_m[:, None]
            B_zero_rate = 1.0 - t[None, :] / n  # linear paydown, no interest
            B = cp.where(is_zero_rate[:, None], B_zero_rate, B)
            B[:, 0] = 1.0

            interest_cum = payment[:, None] * t[None, :] - (1 - B)

            p = pd_g[:, None] * prof[None, :]
            survive = 1 - pd_g

            exp_interest = (
                cp.sum(p[:, 1:n + 1] * interest_cum[:, 1:n + 1], axis=1)
                + survive * interest_cum[:, n]
            )
            exp_loss = cp.sum(p[:, 1:n + 1] * B[:, 1:n + 1] * lgd_g[:, None], axis=1)

            total_margin = principal_g * (exp_interest - exp_loss) - fee_g
            yield_batch = (total_margin / principal_g / (n / 12)) * 100

            total_interest = interest_cum[:, n]
            avg_balance = B[:, :-1].mean(axis=1)
            gross_yield_batch = (total_interest / avg_balance / (n / 12)) * 100

            yield_percent[idx] = cp.asnumpy(yield_batch)
            gross_yield_percent[idx] = cp.asnumpy(gross_yield_batch)

    return yield_percent, gross_yield_percent
##############################################################################################################################
def score_loans(
    df,
    loan_amount_col,
    ltv_col,
    interest_rate_col,
    pd_col,
    term_col,
    curve_type_col=None,
    fee_col=None,
    fee_default=800.0,
    haircut=0.20,
    interest_rate_is_percent="auto",
):
    out = df.copy()

    principal = out[loan_amount_col].to_numpy(dtype=float)
    ltv = out[ltv_col].to_numpy(dtype=float)
    rate_raw = out[interest_rate_col].to_numpy(dtype=float)
    pd_arr = out[pd_col].to_numpy(dtype=float)
    term_months = out[term_col].to_numpy(dtype=int)

    if interest_rate_is_percent == "auto":
        is_percent = rate_raw.max() > 1.0
    else:
        is_percent = bool(interest_rate_is_percent)
    apr = rate_raw / 100.0 if is_percent else rate_raw

    recovery_rate = np.minimum(1.0, (1 - haircut) / ltv)
    lgd = np.maximum(0.0, 1 - recovery_rate)

    if fee_col is not None:
        fee = out[fee_col].to_numpy(dtype=float)
    else:
        fee = np.full(len(out), fee_default, dtype=float)

    curve_type_raw = out[curve_type_col].to_numpy() if curve_type_col is not None else None
    curve_types = _resolve_curve_types(term_months, curve_type_raw)

    yield_percent, gross_yield_percent = _yield_and_gross(
        principal, fee, term_months, pd_arr, apr, lgd, curve_types
    )

    out["mapped_principal"] = principal
    out["mapped_apr"] = apr
    out["mapped_pd"] = pd_arr
    out["mapped_lgd"] = lgd
    out["mapped_fee"] = fee
    out["curve_type_used"] = curve_types
    out["yield_percent"] = yield_percent
    out["gross_yield_percent"] = gross_yield_percent

    return out


def score_portfolio(
    df,
    portfolio_col,
    loan_amount_col=None,
    ltv_col=None,
    apr_col=None,
    pd_col=None,
    term_col=None,
    fee_col=None,
    haircut=0.20,
    defaults=PORTFOLIO_DEFAULTS,
    chunksize=None,
):
    portfolio_type = df[portfolio_col].astype(str).str.strip().str.lower()

    portfolio_type = portfolio_type.replace(_PORTFOLIO_ALIASES)

    unknown = set(portfolio_type.unique()) - set(defaults.keys())
    if unknown:
        raise ValueError(
            f"Unknown portfolio_type value(s) in '{portfolio_col}': {sorted(unknown)}. "
            f"Known types: {sorted(defaults.keys())}. Add them to PORTFOLIO_DEFAULTS first."
        )

    def default_for(field):
        def lookup(p):
            val = defaults[p][field]
            if isinstance(val, (list, tuple, np.ndarray)):
                return np.random.choice(val)
            return val
        return portfolio_type.map(lookup).to_numpy(dtype=float)

    def fill_from_defaults(col, field):
        fallback = default_for(field)
        if col is None or col not in df.columns:
            return fallback
        raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(raw)  # catches NaN AND +/-Inf, not just NaN
        return np.where(missing, fallback, raw)

    principal = fill_from_defaults(loan_amount_col, "principal_default")
    apr = fill_from_defaults(apr_col, "apr_default")
    pd_arr = fill_from_defaults(pd_col, "pd_default")
    term_months = fill_from_defaults(term_col, "term_default").astype(int)
    fee = fill_from_defaults(fee_col, "fee_default")

    # LGD: derive from LTV+haircut where a valid LTV exists, otherwise fall
    # back to the portfolio's flat lgd_default (same haircut logic as before).
    lgd_default = default_for("lgd_default")
    if ltv_col is not None and ltv_col in df.columns:
        ltv_raw = pd.to_numeric(df[ltv_col], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            recovery = np.minimum(1.0, (1 - haircut) / ltv_raw)
        lgd_from_ltv = np.maximum(0.0, 1 - recovery)
        valid = np.isfinite(ltv_raw) & (ltv_raw > 0)  # catches NaN AND +/-Inf
        lgd = np.where(valid, lgd_from_ltv, lgd_default)
    else:
        lgd = lgd_default

    curve_types = portfolio_type.map(lambda p: defaults[p]["curve_type"]).to_numpy()

    if chunksize is not None and len(df) > chunksize:
        yield_percent = np.zeros(len(df), dtype=np.float32)
        gross_yield_percent = np.zeros(len(df), dtype=np.float32)
        for start in range(0, len(df), chunksize):
            end = start + chunksize
            y, g = _yield_and_gross(
                principal[start:end], fee[start:end], term_months[start:end],
                pd_arr[start:end], apr[start:end], lgd[start:end], curve_types[start:end],
            )
            yield_percent[start:end] = y
            gross_yield_percent[start:end] = g
    else:
        yield_percent, gross_yield_percent = _yield_and_gross(
            principal, fee, term_months, pd_arr, apr, lgd, curve_types
        )

    out = df.copy()
    out["portfolio_type_resolved"] = portfolio_type.to_numpy()
    out["mapped_principal"] = principal
    out["mapped_apr"] = apr
    out["mapped_pd"] = pd_arr
    out["mapped_lgd"] = lgd
    out["mapped_term"] = term_months
    out["mapped_fee"] = fee
    out["curve_type_used"] = curve_types
    out["yield_percent"] = yield_percent
    out["gross_yield_percent"] = gross_yield_percent
    return out


def score_loans_chunked(df, *args, chunksize=500_000, **kwargs):
    results = []
    for start in range(0, len(df), chunksize):
        chunk = df.iloc[start:start + chunksize]
        results.append(score_loans(chunk, *args, **kwargs))
    return pd.concat(results, ignore_index=True)


def plot_scored_loans(scored_df, metric="yield_percent", scatter_threshold=2000,
                       apr_col="mapped_apr", pd_col="mapped_pd", save_path=None):
    import matplotlib.pyplot as plt

    apr = scored_df[apr_col].to_numpy() * 100
    pd_vals = scored_df[pd_col].to_numpy() * 100
    values = scored_df[metric].to_numpy()
    finite_mask = np.isfinite(apr) & np.isfinite(pd_vals) & np.isfinite(values)
    n_dropped = (~finite_mask).sum()
    if n_dropped > 0:
        print(f"plot_scored_loans: dropping {n_dropped:,} row(s) with non-finite "
              f"{apr_col}/{pd_col}/{metric} before plotting.")
    apr, pd_vals, values = apr[finite_mask], pd_vals[finite_mask], values[finite_mask]
    n = finite_mask.sum()

    apr_min, apr_max = APR_AXIS_RANGE
    pd_min, pd_max = PD_AXIS_RANGE
    yield_min, yield_max = YIELD_AXIS_RANGE

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    cmap = plt.get_cmap("RdYlGn")

    if n <= scatter_threshold:
        sc = ax.scatter(apr, pd_vals, c=values, cmap=cmap, s=40,
                         edgecolors="black", linewidths=0.3,
                         vmin=yield_min, vmax=yield_max)
        cbar = fig.colorbar(sc, ax=ax)
        mode_note = f"scatter -- {n} individual loans, no binning/interpolation"
    else:
        xedges = np.linspace(apr_min, apr_max, 30)
        yedges = np.linspace(pd_min, pd_max, 30)
        heat, _, _ = np.histogram2d(apr, pd_vals, bins=[xedges, yedges], weights=values)
        counts, _, _ = np.histogram2d(apr, pd_vals, bins=[xedges, yedges])
        with np.errstate(invalid="ignore"):
            heat = heat / counts
        heat[counts == 0] = np.nan
        cmap_bad = cmap.copy()
        cmap_bad.set_bad(color="white")
        im = ax.imshow(heat.T, origin="lower", aspect="auto",
                        extent=[apr_min, apr_max, pd_min, pd_max],
                        cmap=cmap_bad, vmin=yield_min, vmax=yield_max)
        cbar = fig.colorbar(im, ax=ax)
        mode_note = f"binned heatmap -- {n} loans, 30x30 bins"

    ax.set_xlim(apr_min, apr_max)
    ax.set_ylim(pd_min, pd_max)



    label = "Gross Yield (%)" if metric == "gross_yield_percent" else "Net Annualized Yield (%)"


    cbar.set_label(label)
    ax.set_xlabel("APR (%)")
    ax.set_ylabel("PD (%)")


    ax.set_title(f"{state} -- {lei} -- {label} -- {mode_note} -- GPU")




    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    return fig, ax
if __name__ == "__main__":
    df = duck_con.sql(vintage_examined).df()
    print(f"Scoring {len(df):,} real loans on GPU...")
    t0 = time.perf_counter()
    scored = score_portfolio(
        df,
        portfolio_col='portfolio_type',  # <-- the column that HOLDS 'mortgage', not the word itself
        loan_amount_col='loan_amount',
        ltv_col='ltv',
        apr_col='interest_rate_num',
        pd_col='joint_pd_base',
        term_col='loan_term',
        chunksize=500_000,
    )
    cp.cuda.Stream.null.synchronize()
    print(f"  GPU score_portfolio: {time.perf_counter() - t0:.3f}s")
    print(f"n={len(scored)}")
    print(scored[["mapped_apr", "mapped_pd", "mapped_lgd", "curve_type_used",
                    "yield_percent", "gross_yield_percent"]].describe(include="all"))
    plot_scored_loans(scored, metric="yield_percent", save_path=full_path)
    print(f"Saved: {full_path}")
