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
from typing import Optional, Dict, List, Literal
from sklearn.linear_model import LinearRegression, Ridge, Lasso


@dataclass
class PCAResults:
    cov_matrix: Optional[pd.DataFrame] = None
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None
    loadings: Optional[pd.DataFrame] = None
    scores: Optional[pd.DataFrame] = None
    explained_var_ratios: Optional[np.ndarray] = None


@dataclass
class PCARegressionResults:
    selected_tenors: Optional[List[str]] = None
    regression_type: Optional[str] = None
    alpha: Optional[float] = None
    coefficients: Optional[pd.DataFrame] = None
    intercept: Optional[pd.Series] = None
    predicted_scores: Optional[pd.DataFrame] = None
    reduced_matrix: Optional[pd.DataFrame] = None
    reconstructed_processed: Optional[pd.DataFrame] = None
    actual_processed: Optional[pd.DataFrame] = None
    residuals_processed: Optional[pd.DataFrame] = None
    metrics: Optional[Dict] = None


def _revert_processed_frame(data: MarketData, processed_frame: pd.DataFrame):
    """
    Revert a processed frame back to levels using the MarketData preprocessing flags.
    """

    reverted_levels = processed_frame.copy()

    if data.standardize:

        if data.std is None or data.mean is None:
            raise ValueError("Missing mean/std required to invert standardization.")

        reverted_levels = reverted_levels.mul(data.std, axis=1)
        reverted_levels = reverted_levels.add(data.mean, axis=1)

    elif data.demean:

        if data.mean is None:
            raise ValueError("Missing mean required to invert demeaning.")

        reverted_levels = reverted_levels.add(data.mean, axis=1)

    if data.compute_changes:

        original = data.data.loc[
            reverted_levels.index,
            reverted_levels.columns
        ].sort_index()

        first_processed_date = reverted_levels.index[0]
        first_position = original.index.get_loc(first_processed_date)

        if first_position == 0:
            raise ValueError(
                "Cannot invert differencing: no anchor level before first processed date."
            )

        anchor = original.iloc[first_position - 1]

        reverted_levels = reverted_levels.cumsum()
        reverted_levels = reverted_levels.add(anchor, axis=1)

    actual = data.data.loc[
        reverted_levels.index,
        reverted_levels.columns
    ]

    residuals = actual - reverted_levels

    return reverted_levels, actual, residuals



class PCAHedgeRegressor:
    """
    Regress PCA scores on a selected subset of tradable tenors.

    If X is the processed move matrix, L the PCA loadings and F = X L the PCA scores,
    this class estimates a matrix B such that:

        F ≈ H B

    where H contains only the selected hedge tenors. The resulting reduced replication
    matrix is:

        M = B L^T

    so that the full processed surface can be approximated by:

        X_hat = H M = H B L^T
    """

    def __init__(
        self,
        pca: StaticPCA,
        selected_tenors: List[str],
        regression_type: Literal["ols", "ridge", "lasso"] = "ridge",
        alpha: float = 1.0,
        fit_intercept: bool = False,
        max_iter: int = 10000
    ):

        self.pca = pca
        self.selected_tenors = selected_tenors
        self.regression_type = regression_type.lower()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

        self.results: PCARegressionResults = None

    def fit(self):

        if self.pca is None or self.pca.results is None:
            raise ValueError("Fit the PCA before running the regression.")

        if self.pca.data is None or self.pca.data.processed_data is None:
            raise ValueError("Processed data is missing in the PCA object.")

        if self.selected_tenors is None or len(self.selected_tenors) == 0:
            raise ValueError("selected_tenors must contain at least one tenor.")

        processed_data = self.pca.data.processed_data.copy()
        missing_tenors = [tenor for tenor in self.selected_tenors if tenor not in processed_data.columns]

        if missing_tenors:
            raise ValueError(f"Unknown selected tenors: {missing_tenors}")

        selected_moves = processed_data[self.selected_tenors].copy()
        target_scores = self.pca.results.scores.copy()
        pca_loadings = self.pca.results.loadings.copy()

        regression_model = self._build_regression_model()
        regression_model.fit(selected_moves.values, target_scores.values)

        coefficients = np.asarray(regression_model.coef_)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)

        coefficients = coefficients.T

        coefficients_df = pd.DataFrame(
            coefficients,
            index=self.selected_tenors,
            columns=target_scores.columns
        )

        intercept_values = np.asarray(regression_model.intercept_)
        intercept_values = np.atleast_1d(intercept_values).astype(float)

        if intercept_values.size == 1 and target_scores.shape[1] > 1:
            intercept_values = np.repeat(intercept_values, target_scores.shape[1])

        intercept_series = pd.Series(
            intercept_values,
            index=target_scores.columns,
            name="intercept"
        )

        predicted_scores_values = regression_model.predict(selected_moves.values)
        predicted_scores = pd.DataFrame(
            predicted_scores_values,
            index=target_scores.index,
            columns=target_scores.columns
        )

        reduced_matrix = pd.DataFrame(
            coefficients_df.values @ pca_loadings.values.T,
            index=self.selected_tenors,
            columns=pca_loadings.index
        )

        reconstructed_processed = pd.DataFrame(
            predicted_scores.values @ pca_loadings.values.T,
            index=predicted_scores.index,
            columns=pca_loadings.index
        )

        actual_processed = processed_data.loc[
            reconstructed_processed.index,
            reconstructed_processed.columns
        ]

        residuals_processed = actual_processed - reconstructed_processed

        self.results = PCARegressionResults(
            selected_tenors=self.selected_tenors,
            regression_type=self.regression_type,
            alpha=self.alpha,
            coefficients=coefficients_df,
            intercept=intercept_series,
            predicted_scores=predicted_scores,
            reduced_matrix=reduced_matrix,
            reconstructed_processed=reconstructed_processed,
            actual_processed=actual_processed,
            residuals_processed=residuals_processed,
            metrics=reconstruction_metrics(actual_processed, residuals_processed)
        )

    def _build_regression_model(self):

        if self.regression_type == "ols":
            return LinearRegression(fit_intercept=self.fit_intercept)

        if self.regression_type == "ridge":
            return Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept
            )

        if self.regression_type == "lasso":
            return Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter
            )

        raise ValueError(
            f"Unknown regression_type '{self.regression_type}'. "
            "Use 'ols', 'ridge' or 'lasso'."
        )

    def reconstruct(self, revert_processing: bool = False):

        if self.results is None:
            raise RuntimeError("Fit the regression first before trying to reconstruct.")

        if not revert_processing:
            return (
                self.results.reconstructed_processed.copy(),
                self.results.actual_processed.copy(),
                self.results.residuals_processed.copy()
            )

        return _revert_processed_frame(
            self.pca.data,
            self.results.reconstructed_processed.copy()
        )

    def hedge_ratios_for_node(self, node: str) -> pd.Series:

        if self.results is None:
            raise RuntimeError("Fit the regression first before requesting hedge ratios.")

        if node not in self.results.reduced_matrix.columns:
            raise ValueError(f"Unknown node '{node}'.")

        return self.results.reduced_matrix[node].copy()

    def hedge_weights_from_sensitivities(self, sensitivities: pd.Series) -> pd.Series:

        if self.results is None:
            raise RuntimeError("Fit the regression first before projecting sensitivities.")

        if not isinstance(sensitivities, pd.Series):
            raise TypeError("sensitivities must be provided as a pandas Series.")

        missing_nodes = [node for node in self.results.reduced_matrix.columns if node not in sensitivities.index]
        if missing_nodes:
            raise ValueError(
                "Missing nodes in sensitivities input. "
                f"Example missing nodes: {missing_nodes[:10]}"
            )

        ordered_sensitivities = sensitivities.loc[self.results.reduced_matrix.columns]

        hedge_weights = self.results.reduced_matrix.values @ ordered_sensitivities.values

        return pd.Series(
            hedge_weights,
            index=self.results.reduced_matrix.index,
            name="hedge_weights"
        )


class RollingPCA:
    def fit_regressions(
        self,
        selected_tenors: List[str],
        regression_type: Literal["ols", "ridge", "lasso"] = "ridge",
        alpha: float = 1.0,
        fit_intercept: bool = False,
        max_iter: int = 10000
    ) -> List[PCAHedgeRegressor]:

        if not self.pcas_results:
            raise ValueError("Run fit() first before fitting rolling regressions.")

        self.regressions_results = []

        for pca in self.pcas_results:
            regression = PCAHedgeRegressor(
                pca=pca,
                selected_tenors=selected_tenors,
                regression_type=regression_type,
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=max_iter
            )
            regression.fit()
            self.regressions_results.append(regression)

        return self.regressions_results
