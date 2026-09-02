"""Simulate minute-by-minute synthetic yield curves with a 3-factor PPCA model.

The model is applied to yield changes:

    delta_y_t = W z_t + epsilon_t,

where z_t ~ N(0, I_3), the columns of W describe level, slope and
curvature shocks, and epsilon_t is independent Gaussian observation noise.
Yields are then obtained by cumulatively summing the changes.

Rates are represented as decimals (0.04 means 4%). Volatility parameters are
annualised and expressed in basis points.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


TENORS = np.array([1, 2, 3, 5, 7, 10, 12, 15, 20, 30], dtype=float)
COLUMNS = ["1y", "2y", "3y", "5y", "7y", "10y", "12y", "15y", "20y", "30y"]

# An illustrative initial curve. Replace this with any curve you prefer.
# Values are decimal yields, so 0.0400 is 4.00%.
DEFAULT_INITIAL_CURVE = np.array(
    [0.0400, 0.0390, 0.0385, 0.0380, 0.0382,
     0.0390, 0.0395, 0.0400, 0.0410, 0.0420],
    dtype=float,
)


def _period_start(period: str, end: pd.Timestamp) -> pd.Timestamp:
    """Return the start timestamp implied by strings such as 30min, 3m or 5y.

    Syntax is deliberately unambiguous: ``min`` means minutes and ``m`` means
    calendar months. Other supported units are h, d, w and y.
    """
    match = re.fullmatch(r"\s*(\d+)\s*(min|h|d|w|m|y)\s*", period.lower())
    if match is None:
        raise ValueError(
            "period must look like '30min', '12h', '10d', '2w', '3m', or '5y'"
        )

    n = int(match.group(1))
    unit = match.group(2)
    if n <= 0:
        raise ValueError("period must be strictly positive")

    if unit == "min":
        return end - pd.Timedelta(minutes=n)
    if unit == "h":
        return end - pd.Timedelta(hours=n)
    if unit == "d":
        return end - pd.Timedelta(days=n)
    if unit == "w":
        return end - pd.Timedelta(weeks=n)
    if unit == "m":
        return end - pd.DateOffset(months=n)
    return end - pd.DateOffset(years=n)


def make_factor_loadings(tenors: np.ndarray = TENORS) -> pd.DataFrame:
    """Construct level, slope and curvature loading shapes.

    The maturity coordinate is log tenor, rescaled to [-1, 1]. A positive
    slope shock raises long rates relative to short rates. A positive curvature
    shock raises the belly relative to the two ends. Curvature is residualised
    against level and slope so the three shapes are distinct.

    Each column is scaled to have a maximum absolute loading of one. Annualised
    factor volatilities therefore control the largest tenor exposure to each
    factor rather than the Euclidean norm of the loading vector.
    """
    tenors = np.asarray(tenors, dtype=float)
    if tenors.ndim != 1 or len(tenors) < 3 or np.any(tenors <= 0):
        raise ValueError("tenors must be a one-dimensional array of positive values")

    log_tenor = np.log(tenors)
    x = 2.0 * (log_tenor - log_tenor.min()) / np.ptp(log_tenor) - 1.0

    level = np.ones_like(x)
    slope = x.copy()
    raw_curvature = 1.0 - x**2

    # Remove the projections of curvature onto level and slope.
    base = np.column_stack((level, slope))
    curvature = raw_curvature - base @ np.linalg.lstsq(
        base, raw_curvature, rcond=None
    )[0]

    loadings = np.column_stack((level, slope, curvature))
    loadings /= np.max(np.abs(loadings), axis=0)

    index = [f"{tenor:g}y" for tenor in tenors]
    return pd.DataFrame(
        loadings,
        index=index,
        columns=["level", "slope", "curvature"],
    )


def _coerce_initial_curve(
    initial_curve: Sequence[float] | Mapping[str, float] | pd.Series | None,
) -> np.ndarray:
    if initial_curve is None:
        values = DEFAULT_INITIAL_CURVE.copy()
    elif isinstance(initial_curve, (Mapping, pd.Series)):
        missing = [column for column in COLUMNS if column not in initial_curve]
        if missing:
            raise ValueError(f"initial_curve is missing tenors: {missing}")
        values = np.array([initial_curve[column] for column in COLUMNS], dtype=float)
    else:
        values = np.asarray(initial_curve, dtype=float)

    if values.shape != (len(COLUMNS),):
        raise ValueError(f"initial_curve must contain exactly {len(COLUMNS)} rates")
    if not np.all(np.isfinite(values)):
        raise ValueError("initial_curve must contain only finite values")
    return values


def simulate_yield_curve(
    period: str,
    *,
    end: str | pd.Timestamp | None = None,
    initial_curve: Sequence[float] | Mapping[str, float] | pd.Series | None = None,
    factor_vols_bp: Sequence[float] = (70.0, 45.0, 30.0),
    noise_vol_bp: float = 10.0,
    weekdays_only: bool = True,
    timezone: str = "UTC",
    seed: int | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Simulate a PPCA yield curve ending at the current time.

    Parameters
    ----------
    period:
        History length. ``m`` means calendar months and ``min`` means minutes;
        examples include ``"3m"`` and ``"5y"``.
    end:
        Final timestamp. If omitted, the current time is used. It is useful to
        specify this when reproducibility must include an identical index.
    initial_curve:
        Ten starting yields in the order given by ``COLUMNS``, or a mapping
        keyed by those column names. Rates are decimals. Defaults to the
        illustrative curve above.
    factor_vols_bp:
        Annualised level, slope and curvature volatilities in basis points.
    noise_vol_bp:
        Annualised tenor-specific PPCA noise volatility in basis points.
    weekdays_only:
        If True, retain all 1-minute observations Monday through Friday, 24
        hours per day. If False, simulate every calendar minute, including
        weekends.
    timezone:
        Timezone for the DatetimeIndex. Defaults to UTC.
    seed:
        NumPy random seed. Supply an integer for reproducible shocks.
    chunk_size:
        Number of changes generated at once. Chunking keeps multi-year runs
        from needing several large temporary arrays.

    Returns
    -------
    pandas.DataFrame
        Decimal yields with a timezone-aware, 1-minute DatetimeIndex named
        ``datestamp`` and columns 1y through 30y.

    Notes
    -----
    The default annualisation uses 252 * 24 * 60 tradable minutes when weekends
    are excluded, and 365.25 * 24 * 60 minutes for a 24/7 series. The process is
    a random walk, as requested: it has neither mean reversion nor a positivity
    constraint.
    """
    factor_vols_bp = np.asarray(factor_vols_bp, dtype=float)
    if (
        factor_vols_bp.shape != (3,)
        or not np.all(np.isfinite(factor_vols_bp))
        or np.any(factor_vols_bp < 0)
    ):
        raise ValueError("factor_vols_bp must contain three non-negative values")
    if not np.isfinite(noise_vol_bp) or noise_vol_bp < 0:
        raise ValueError("noise_vol_bp must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be strictly positive")

    if end is None:
        end_ts = pd.Timestamp.now(tz=timezone).floor("min")
    else:
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(timezone)
        else:
            end_ts = end_ts.tz_convert(timezone)
        end_ts = end_ts.floor("min")

    start_ts = _period_start(period, end_ts)
    index = pd.date_range(start=start_ts, end=end_ts, freq="min")
    if weekdays_only:
        index = index[index.dayofweek < 5]
    index.name = "datestamp"
    if len(index) == 0:
        raise ValueError("the selected period contains no timestamps on the chosen grid")

    initial = _coerce_initial_curve(initial_curve)
    loadings = make_factor_loadings(TENORS).to_numpy()

    annual_minutes = (252.0 if weekdays_only else 365.25) * 24.0 * 60.0
    sqrt_dt = np.sqrt(1.0 / annual_minutes)
    bp_to_decimal = 1.0e-4

    # W has units of decimal yield change per one-standard-deviation factor.
    w = loadings * (factor_vols_bp * bp_to_decimal * sqrt_dt)
    noise_sd = noise_vol_bp * bp_to_decimal * sqrt_dt

    rng = np.random.default_rng(seed)
    rates = np.empty((len(index), len(COLUMNS)), dtype=float)
    rates[0] = initial

    current = initial.copy()
    write_at = 1
    while write_at < len(index):
        n = min(chunk_size, len(index) - write_at)
        latent_factors = rng.standard_normal((n, 3))
        changes = latent_factors @ w.T
        if noise_sd > 0:
            changes += rng.standard_normal((n, len(COLUMNS))) * noise_sd

        block = current + np.cumsum(changes, axis=0)
        rates[write_at : write_at + n] = block
        current = block[-1]
        write_at += n

    return pd.DataFrame(rates, index=index, columns=COLUMNS)


if __name__ == "__main__":
    # A small reproducible example. Replace "5d" with "3m" or "5y" as needed.
    curves = simulate_yield_curve("5d", seed=42)
    print(curves.head())
    print(curves.tail())
    print(f"shape = {curves.shape}")
