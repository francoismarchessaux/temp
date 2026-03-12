import numpy as np
import pandas as pd
from itertools import permutations

from utils import *
from market_data import *
from covariance_estimators import *

import seaborn as sns
import matplotlib.pyplot as plt

from dataclasses import dataclass
from IPython.display import display
from typing import Optional, Dict, List, Literal, Union

@dataclass
class PCAHedgePnLResults:
    price_moves: Optional[pd.DataFrame] = None
    true_pnl_contributions: Optional[pd.DataFrame] = None
    pca_pnl_contributions: Optional[pd.DataFrame] = None
    true_pnl: Optional[pd.Series] = None
    pca_pnl: Optional[pd.Series] = None
    comparison: Optional[pd.DataFrame] = None
    pca_sensi: Optional[pd.Series] = None
    projection_matrix: Optional[pd.DataFrame] = None


def _extract_raw_market_data(market_data: Union[MarketData, pd.DataFrame]) -> pd.DataFrame:
    """
    Return raw market levels as a DataFrame sorted by date.

    Parameters
    ----------
    market_data : MarketData or pd.DataFrame
        - If MarketData, uses market_data.data
        - If DataFrame, uses it directly
    """

    if isinstance(market_data, MarketData):
        if market_data.data is None:
            raise ValueError("MarketData.data is empty. Call load() first.")
        raw_market_data = market_data.data.copy()
    elif isinstance(market_data, pd.DataFrame):
        raw_market_data = market_data.copy()
    else:
        raise TypeError(
            "market_data must be either a MarketData object or a pandas DataFrame."
        )

    if not isinstance(raw_market_data.index, pd.DatetimeIndex):
        raw_market_data.index = pd.to_datetime(raw_market_data.index)

    raw_market_data = raw_market_data.sort_index()

    return raw_market_data


def compute_forward_price_moves(
    market_data: Union[MarketData, pd.DataFrame],
    horizon_days: int = 5
) -> pd.DataFrame:
    """
    Compute forward price moves P(t+h) - P(t), expressed in the same units as the
    input market data. For rates curves already stored in bp, the output is in bp.

    The result is indexed by the start date t of the holding period.
    """

    if horizon_days <= 0:
        raise ValueError("horizon_days must be strictly positive.")

    raw_market_data = _extract_raw_market_data(market_data)

    forward_price_moves = raw_market_data.shift(-horizon_days) - raw_market_data
    forward_price_moves = forward_price_moves.dropna(how="any")

    return forward_price_moves


def compute_pnl_from_price_moves(
    price_moves: pd.DataFrame,
    sensitivities: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute tenor-by-tenor and total PNL from market moves and sensitivities.

    Parameters
    ----------
    price_moves : pd.DataFrame
        Market moves by date and tenor.

    sensitivities : pd.Series
        Sensitivity vector indexed by tenor.

    Returns
    -------
    pnl_contributions : pd.DataFrame
        Tenor-level PNL contributions.

    total_pnl : pd.Series
        Sum of tenor-level contributions by date.
    """

    if not isinstance(sensitivities, pd.Series):
        raise TypeError("sensitivities must be a pandas Series indexed by tenor.")

    sensitivities_aligned = sensitivities.copy()
    sensitivities_aligned.index = sensitivities_aligned.index.astype(str)

    price_moves_aligned = price_moves.reindex(columns=sensitivities_aligned.index).fillna(0.0)

    pnl_contributions = price_moves_aligned.mul(sensitivities_aligned, axis=1)
    total_pnl = pnl_contributions.sum(axis=1)

    return pnl_contributions, total_pnl


def compare_true_and_pca_pnl(
    market_data: Union[MarketData, pd.DataFrame],
    true_sensi_position: pd.Series,
    pca_sensi: pd.Series,
    horizon_days: int = 5
) -> PCAHedgePnLResults:
    """
    Compare historical true PNL against PCA hedge PNL over a forward holding period.

    Methodology implemented:
        1. PCA Move = P(t+h) - P(t) on PCA hedge tenors
        2. Sensi = PCA hedge sensitivities
        3. PCA PNL = sum(PCA Move x PCA Sensi)
        4. True PNL = sum(Delta price by tenor x true sensitivity position)
        5. Diff = True PNL - PCA PNL
        6. Diff % = Diff / True PNL

    Notes
    -----
    - The comparison is done for every date t such that t+h exists in the sample.
    - The result is dated at t, i.e. the start of the forward holding period.
    - When True PNL is zero, Diff % is set to NaN to avoid division by zero.
    """

    price_moves = compute_forward_price_moves(
        market_data=market_data,
        horizon_days=horizon_days
    )

    true_pnl_contributions, true_pnl = compute_pnl_from_price_moves(
        price_moves=price_moves,
        sensitivities=true_sensi_position
    )

    pca_price_moves = price_moves.reindex(columns=pca_sensi.index).fillna(0.0)
    pca_pnl_contributions, pca_pnl = compute_pnl_from_price_moves(
        price_moves=pca_price_moves,
        sensitivities=pca_sensi
    )

    pnl_difference = true_pnl - pca_pnl
    pnl_difference_pct = pnl_difference.divide(true_pnl.replace(0.0, np.nan))

    comparison = pd.DataFrame({
        "True PNL": true_pnl,
        "PCA PNL": pca_pnl,
        "Diff": pnl_difference,
        "Diff %": pnl_difference_pct
    })

    return PCAHedgePnLResults(
        price_moves=price_moves,
        true_pnl_contributions=true_pnl_contributions,
        pca_pnl_contributions=pca_pnl_contributions,
        true_pnl=true_pnl,
        pca_pnl=pca_pnl,
        comparison=comparison,
        pca_sensi=pca_sensi
    )


def backtest_pca_hedge(
    pca: StaticPCA,
    true_sensi_position: pd.Series,
    hedge_tenors: list[str],
    horizon_days: int = 5
) -> PCAHedgePnLResults:
    """
    End-to-end helper:
        1. Build projection matrix from PCA loadings to hedge tenors
        2. Compute PCA hedge sensitivities
        3. Compare historical True PNL vs PCA PNL

    Parameters
    ----------
    pca : StaticPCA
        Fitted PCA object.

    true_sensi_position : pd.Series
        Original desk sensitivity vector indexed by full tenor universe.

    hedge_tenors : list[str]
        Tenors used as PCA hedge pillars.

    horizon_days : int
        Forward holding period in business-day rows. Default = 5.
    """

    if pca.results is None or pca.results.loadings is None:
        raise ValueError("PCA must be fitted before running the hedge backtest.")

    projection_matrix = project_pca_loadings_on_hedge_tenors(
        loadings=pca.results.loadings,
        hedge_tenors=hedge_tenors
    )

    pca_sensi = compute_pca_hedge(
        sensi=true_sensi_position,
        proj_matrix=projection_matrix
    )

    pnl_results = compare_true_and_pca_pnl(
        market_data=pca.data,
        true_sensi_position=true_sensi_position,
        pca_sensi=pca_sensi,
        horizon_days=horizon_days
    )

    pnl_results.projection_matrix = projection_matrix
    pnl_results.pca_sensi = pca_sensi

    return pnl_results


import pandas as pd

from pca_engines_with_pnl import (
    StaticPCA,
    backtest_pca_hedge,
    project_pca_loadings_on_hedge_tenors,
    compute_pca_hedge,
    compare_true_and_pca_pnl
)
from market_data import MarketData, CurveInfo
from covariance_estimators import CovarianceEstimator

# Example: after market_data.load() and market_data.preprocess(...)
cov_estimator = CovarianceEstimator(method="sample")

pca = StaticPCA(
    data=market_data,
    n_components=3,
    cov_estimator=cov_estimator
)
pca.fit()

# Full desk sensitivity vector on the whole tenor universe
true_sensi_position = pd.Series(
    {
        "2Y": 0.0,
        "3Y": 0.0,
        "4Y": 0.0,
        "5Y": 125000.0,
        "6Y": 0.0,
        "7Y": -40000.0,
        "8Y": 0.0,
        "9Y": 0.0,
        "10Y": 210000.0,
        "11Y": 0.0,
        "12Y": 0.0,
        "13Y": 0.0,
        "14Y": 0.0,
        "15Y": -150000.0,
        "20Y": 90000.0,
        "25Y": 0.0,
        "30Y": -120000.0,
        "40Y": 0.0,
        "50Y": 0.0
    }
)

hedge_tenors = ["5Y", "10Y", "30Y"]

pnl_results = backtest_pca_hedge(
    pca=pca,
    true_sensi_position=true_sensi_position,
    hedge_tenors=hedge_tenors,
    horizon_days=5
)

print("Projected PCA hedge sensitivities:")
print(pnl_results.pca_sensi)

print("\nPNL comparison:")
print(pnl_results.comparison.head())

print("\nProjection matrix:")
print(pnl_results.projection_matrix)
