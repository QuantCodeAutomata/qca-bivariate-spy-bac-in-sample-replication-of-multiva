"""Data loading interface for HEAVY vs GARCH replication.

Attempts to fetch daily OHLCV data from the Massive API for SPY and BAC
(2001-02-01 to 2009-12-31).  If the Massive API is unavailable, the
MASSIVE_TOKEN environment variable is unset, or the returned series is
shorter than 500 rows, falls back transparently to synthetic data from
:mod:`data.generate_synthetic`.

Functions
---------
get_bivariate_data()
    Bivariate SPY-BAC dataset (~2242 trading days).
get_ten_asset_data(assets_list)
    Multi-asset dataset for an arbitrary list of tickers.

In both cases the returned dict always has keys:
    'returns' : np.ndarray (T, k)
    'P'       : np.ndarray (T, k, k)  - outer products r_t r_t'
    'V'       : np.ndarray (T, k, k)  - realized covariance matrices
    'dates'   : pd.DatetimeIndex of length T

Note on Massive API
-------------------
The client is instantiated as::

    from massive import RESTClient
    client = RESTClient(api_key=os.getenv('MASSIVE_TOKEN'))

TAQ intraday data for 2001-2009 is *not* available through the Massive API
(which covers only recent data), so the synthetic fallback is the normal
code path.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.generate_synthetic import generate_bivariate_data, generate_multiasset_data

logger = logging.getLogger(__name__)

_START = "2001-02-01"
_END = "2009-12-31"
_BIVARIATE_ASSETS = ["SPY", "BAC"]
_MIN_ROWS = 500      # fall back if fewer rows than this are fetched
_FULL_N_DAYS = len(pd.bdate_range(start=_START, end=_END))  # ~2242


# ---------------------------------------------------------------------------
# Internal: Massive API fetch (best-effort)
# ---------------------------------------------------------------------------

def _fetch_massive_daily(ticker: str) -> pd.DataFrame | None:
    """Attempt to fetch daily OHLCV bars from the Massive API.

    Returns None on any failure (missing token, import error, HTTP error,
    insufficient data).

    Parameters
    ----------
    ticker:
        Ticker symbol, e.g. 'SPY'.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with at minimum a 'close' column and a DatetimeIndex,
        or None if data is unavailable.
    """
    token = os.getenv("MASSIVE_TOKEN")
    if not token:
        logger.debug("MASSIVE_TOKEN not set; skipping Massive API fetch.")
        return None
    try:
        from massive import RESTClient  # type: ignore[import]
        client = RESTClient(api_key=token)
        bars = client.get_daily_bars(
            ticker=ticker,
            start=_START,
            end=_END,
        )
        df = pd.DataFrame(bars)
        if df.empty or len(df) < _MIN_ROWS:
            logger.warning(
                "Massive API returned %d rows for %s (< %d); "
                "falling back to synthetic.",
                len(df),
                ticker,
                _MIN_ROWS,
            )
            return None
        df.index = pd.to_datetime(df.index)
        logger.info("Massive API: fetched %d rows for %s.", len(df), ticker)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Massive API fetch failed for %s: %s", ticker, exc)
        return None


def _returns_from_ohlcv(df: pd.DataFrame) -> np.ndarray:
    """Compute log-returns from daily close prices.

    Parameters
    ----------
    df:
        DataFrame with a 'close' column.

    Returns
    -------
    np.ndarray
        1-D array of log-returns (length len(df) - 1).
    """
    closes = df["close"].values.astype(float)
    return np.diff(np.log(closes))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_bivariate_data(seed: int = 42) -> dict:
    """Load bivariate SPY-BAC dataset for 2001-02-01 to 2009-12-31.

    Tries the Massive API first; falls back to synthetic data from
    :func:`data.generate_synthetic.generate_bivariate_data` when the API
    is unavailable or returns insufficient data.

    Parameters
    ----------
    seed:
        Random seed used only if synthetic fallback is triggered.

    Returns
    -------
    dict
        ``'returns'`` (T, 2), ``'P'`` (T, 2, 2), ``'V'`` (T, 2, 2),
        ``'dates'`` (pd.DatetimeIndex).
    """
    spy_df = _fetch_massive_daily("SPY")
    bac_df = _fetch_massive_daily("BAC")

    if spy_df is not None and bac_df is not None:
        # Align on common dates
        common = spy_df.index.intersection(bac_df.index)
        if len(common) >= _MIN_ROWS:
            spy_r = _returns_from_ohlcv(spy_df.loc[common])
            bac_r = _returns_from_ohlcv(bac_df.loc[common])
            T = min(len(spy_r), len(bac_r))
            dates = pd.bdate_range(start=common[1], periods=T)
            returns = np.column_stack([spy_r[:T], bac_r[:T]])
            P = np.einsum("ti,tj->tij", returns, returns)
            # Without intraday data, use P as a crude realized-cov proxy
            V = P.copy()
            logger.info("Loaded REAL bivariate data from Massive API: T=%d.", T)
            return {"returns": returns, "P": P, "V": V, "dates": dates}

    # Synthetic fallback (expected normal path for 2001-2009 TAQ data)
    logger.info(
        "Using SYNTHETIC bivariate data (Massive API unavailable for 2001-2009)."
    )
    data = generate_bivariate_data(n_days=_FULL_N_DAYS, seed=seed)
    print(
        f"[fetch_data] Synthetic bivariate SPY-BAC: T={_FULL_N_DAYS}, "
        f"{data['dates'][0].date()} to {data['dates'][-1].date()}"
    )
    return data


def get_ten_asset_data(assets_list: list | None = None, seed: int = 42) -> dict:
    """Load a multi-asset dataset for the given tickers.

    Tries the Massive API for each ticker; if any fetch fails or returns
    insufficient data, falls back entirely to synthetic data for all assets.

    Parameters
    ----------
    assets_list:
        List of ticker strings.  Defaults to a ten-asset equity basket
        (BAC, JPM, IBM, MSFT, XOM, AA, AXP, DD, GE, KO) when None.
    seed:
        Random seed used only if synthetic fallback is triggered.

    Returns
    -------
    dict
        ``'returns'`` (T, k), ``'P'`` (T, k, k), ``'V'`` (T, k, k),
        ``'dates'`` (pd.DatetimeIndex).
    """
    if assets_list is None:
        assets_list = ["BAC", "JPM", "IBM", "MSFT", "XOM",
                       "AA", "AXP", "DD", "GE", "KO"]

    dfs = {ticker: _fetch_massive_daily(ticker) for ticker in assets_list}
    available = {t: df for t, df in dfs.items() if df is not None}

    if len(available) == len(assets_list):
        common = None
        for df in available.values():
            common = df.index if common is None else common.intersection(df.index)
        if common is not None and len(common) >= _MIN_ROWS:
            all_r = []
            for ticker in assets_list:
                all_r.append(_returns_from_ohlcv(available[ticker].loc[common]))
            T = min(len(arr) for arr in all_r)
            returns = np.column_stack([arr[:T] for arr in all_r])
            dates = pd.bdate_range(start=common[1], periods=T)
            P = np.einsum("ti,tj->tij", returns, returns)
            V = P.copy()
            logger.info(
                "Loaded REAL multi-asset data from Massive API: T=%d, k=%d.",
                T, len(assets_list),
            )
            return {"returns": returns, "P": P, "V": V, "dates": dates}

    logger.info(
        "Using SYNTHETIC multi-asset data for %s "
        "(Massive API unavailable for 2001-2009).",
        assets_list,
    )
    data = generate_multiasset_data(
        assets=assets_list, n_days=_FULL_N_DAYS, seed=seed
    )
    print(
        f"[fetch_data] Synthetic {len(assets_list)}-asset data: T={_FULL_N_DAYS}, "
        f"{data['dates'][0].date()} to {data['dates'][-1].date()}"
    )
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d2 = get_bivariate_data()
    print("Bivariate returns:", d2["returns"].shape,
          "V:", d2["V"].shape)
    d10 = get_ten_asset_data()
    print("Ten-asset returns:", d10["returns"].shape,
          "V:", d10["V"].shape)
