from __future__ import annotations

import io
import itertools
import math
import os
import sys
import tempfile
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Compatibility shim for environments where pyJade is unavailable.
# This must be defined before importing market_data / utils because those modules import
# pyJade at import time.
# --------------------------------------------------------------------------------------
try:
    from pyJade.pearl import pearl_service  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    pyjade_module = types.ModuleType("pyJade")
    pearl_module = types.ModuleType("pyJade.pearl")

    def pearl_service():
        class DummyPearlService:
            def __getattr__(self, attribute_name: str):
                raise ModuleNotFoundError(
                    "pyJade is not available in this environment. "
                    "Use file-upload mode in the Streamlit dashboard."
                )

        return DummyPearlService()

    pearl_module.pearl_service = pearl_service
    pyjade_module.pearl = pearl_module
    sys.modules["pyJade"] = pyjade_module
    sys.modules["pyJade.pearl"] = pearl_module

from market_data import CurveInfo, MarketData
from covariance_estimators import CovarianceEstimator
from pca_engines_with_pnl import (
    StaticPCA,
    backtest_pca_hedge,
    compare_true_and_pca_pnl,
    compute_pca_hedge,
    project_pca_loadings_on_hedge_tenors,
)
from utils import reconstruction_metrics


# --------------------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="PCA Hedge Dashboard", layout="wide")

DEFAULT_TENORS = [
    "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "11Y",
    "12Y", "13Y", "14Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"
]

DEFAULT_HEDGE_SEARCH_UNIVERSE = ["2Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y", "50Y"]
DEFAULT_PLOT_TENORS = ["2Y", "5Y", "10Y", "30Y", "50Y"]
DEFAULT_COVARIANCE_CHOICES = ["sample", "ledoit_wolf", "OAS", "ewma_hl_20", "ewma_hl_60"]
DEFAULT_PREPROCESSING_CHOICES = [
    "chg=1|demean=1|std=0",
    "chg=1|demean=0|std=1",
    "chg=0|demean=1|std=0",
    "chg=0|demean=0|std=1",
]

DEFAULT_RANKING_WEIGHTS = {
    "mean_abs_diff_ratio": 0.35,
    "rmse_diff": 0.20,
    "p95_abs_diff_pct": 0.15,
    "pnl_correlation_penalty": 0.15,
    "loading_condition_number": 0.05,
    "explained_var_penalty": 0.10,
}


@dataclass
class DashboardConfig:
    source_path: str
    curve_info: CurveInfo
    start_date: str
    end_date: str
    rates_tenors_universe: List[str]
    true_sensitivity_vector: pd.Series
    preprocessing_grid: List[Dict]
    covariance_grid: List[Dict]
    n_components_grid: List[int]
    hedge_search_universe: List[str]
    manual_hedge_sets: Dict[int, List[Tuple[str, ...]]]
    add_all_combinations_up_to_k: int
    max_hedge_combinations_per_k: int
    pnl_horizon_days: int
    rolling_window_rows_grid: List[int]
    rolling_step_rows: int
    plot_sample_tenors: List[str]
    ranking_weights: Dict[str, float]


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------
def ensure_datetime_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    converted_dataframe = dataframe.copy()
    if not isinstance(converted_dataframe.index, pd.DatetimeIndex):
        converted_dataframe.index = pd.to_datetime(converted_dataframe.index)
    return converted_dataframe.sort_index()


def save_uploaded_file(uploaded_file) -> str:
    file_suffix = Path(uploaded_file.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def parse_preprocessing_choice(choice_label: str) -> Dict:
    return {
        "compute_changes": "chg=1" in choice_label,
        "demean": "demean=1" in choice_label,
        "standardize": "std=1" in choice_label,
    }


def preprocessing_label(preprocessing_spec: Dict) -> str:
    return (
        f"chg={int(preprocessing_spec['compute_changes'])}|"
        f"demean={int(preprocessing_spec['demean'])}|"
        f"std={int(preprocessing_spec['standardize'])}"
    )


def parse_covariance_choice(choice_label: str) -> Dict:
    if choice_label == "sample":
        return {"method": "sample"}
    if choice_label == "ledoit_wolf":
        return {"method": "ledoit_wolf"}
    if choice_label == "OAS":
        return {"method": "OAS"}
    if choice_label == "ewma_hl_20":
        return {"method": "ewma", "halflife": 20}
    if choice_label == "ewma_hl_60":
        return {"method": "ewma", "halflife": 60}
    if choice_label == "graphical_lasso":
        return {"method": "graphical_lasso", "alpha_lasso": 0.01, "max_iter": 100}
    if choice_label == "graphical_lasso_cv":
        return {"method": "graphical_lasso_cv", "max_iter": 100}
    raise ValueError(f"Unsupported covariance choice: {choice_label}")


def covariance_label(covariance_spec: Dict) -> str:
    method = covariance_spec["method"]
    if method == "ewma":
        if covariance_spec.get("halflife") is not None:
            return f"ewma_hl={covariance_spec['halflife']}"
        if covariance_spec.get("alpha_ewma") is not None:
            return f"ewma_alpha={covariance_spec['alpha_ewma']}"
    return str(method)


def parse_tuple_text(tuple_text: str) -> Dict[int, List[Tuple[str, ...]]]:
    parsed_manual_sets: Dict[int, List[Tuple[str, ...]]] = {}
    cleaned_text = tuple_text.strip()
    if not cleaned_text:
        return parsed_manual_sets

    for raw_block in cleaned_text.split(";"):
        block = raw_block.strip()
        if not block:
            continue
        hedge_tenors = tuple(item.strip() for item in block.split(",") if item.strip())
        if not hedge_tenors:
            continue
        parsed_manual_sets.setdefault(len(hedge_tenors), []).append(hedge_tenors)

    return parsed_manual_sets


def make_default_sensitivity_dataframe(tenors: List[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "tenor": tenors,
        "sensitivity": [0.0 for _ in tenors],
    })


def dataframe_to_sensitivity_series(sensitivity_dataframe: pd.DataFrame) -> pd.Series:
    cleaned_dataframe = sensitivity_dataframe.copy()
    cleaned_dataframe["tenor"] = cleaned_dataframe["tenor"].astype(str)
    cleaned_dataframe["sensitivity"] = pd.to_numeric(cleaned_dataframe["sensitivity"], errors="coerce").fillna(0.0)
    return cleaned_dataframe.set_index("tenor")["sensitivity"].astype(float)


# --------------------------------------------------------------------------------------
# Core PCA study functions
# --------------------------------------------------------------------------------------
def make_base_market_data(config: DashboardConfig) -> MarketData:
    market_data = MarketData(
        curve_info=config.curve_info,
        source=config.source_path,
        start_date=config.start_date,
        end_date=config.end_date,
        rates_tenors_universe=config.rates_tenors_universe,
    )
    market_data.load()
    return market_data


def clone_market_data_with_preprocessing(
    raw_market_data: pd.DataFrame,
    config: DashboardConfig,
    preprocessing_spec: Dict,
) -> MarketData:
    cloned_market_data = MarketData(
        curve_info=config.curve_info,
        source=config.source_path,
        start_date=config.start_date,
        end_date=config.end_date,
        rates_tenors_universe=config.rates_tenors_universe,
    )
    cloned_market_data.data = raw_market_data.copy()
    cloned_market_data.preprocess(**preprocessing_spec)
    return cloned_market_data


def build_covariance_estimator(covariance_spec: Dict) -> CovarianceEstimator:
    return CovarianceEstimator(**covariance_spec)


def fit_static_pca_model(
    raw_market_data: pd.DataFrame,
    config: DashboardConfig,
    preprocessing_spec: Dict,
    covariance_spec: Dict,
    n_components: int,
) -> Tuple[MarketData, StaticPCA]:
    processed_market_data = clone_market_data_with_preprocessing(
        raw_market_data=raw_market_data,
        config=config,
        preprocessing_spec=preprocessing_spec,
    )

    covariance_estimator = build_covariance_estimator(covariance_spec)
    pca_model = StaticPCA(
        data=processed_market_data,
        n_components=n_components,
        cov_estimator=covariance_estimator,
    )
    pca_model.fit()
    return processed_market_data, pca_model


def build_candidate_hedge_sets(
    n_components: int,
    hedge_universe: List[str],
    manual_sets_dictionary: Dict[int, List[Tuple[str, ...]]],
    add_all_combinations: bool,
    max_combinations: int,
) -> List[Tuple[str, ...]]:
    candidate_hedge_sets: List[Tuple[str, ...]] = []

    for hedge_tuple in manual_sets_dictionary.get(n_components, []):
        if len(hedge_tuple) == n_components:
            candidate_hedge_sets.append(tuple(hedge_tuple))

    if add_all_combinations:
        generated_combinations = list(itertools.combinations(hedge_universe, n_components))
        if len(generated_combinations) > max_combinations:
            raise ValueError(
                f"There are {len(generated_combinations)} hedge combinations for k={n_components}, "
                f"which exceeds max_hedge_combinations_per_k={max_combinations}."
            )
        candidate_hedge_sets.extend(generated_combinations)

    deduplicated_hedge_sets: List[Tuple[str, ...]] = []
    seen_hedge_sets = set()
    for hedge_tuple in candidate_hedge_sets:
        if hedge_tuple not in seen_hedge_sets:
            deduplicated_hedge_sets.append(hedge_tuple)
            seen_hedge_sets.add(hedge_tuple)

    return deduplicated_hedge_sets


def compute_loading_condition_number(pca_model: StaticPCA, hedge_tenors: Tuple[str, ...]) -> float:
    loading_matrix = pca_model.results.loadings.loc[list(hedge_tenors)].values
    try:
        return float(np.linalg.cond(loading_matrix))
    except Exception:
        return np.nan


def summarize_pnl_quality(pnl_results) -> Dict[str, float]:
    pnl_comparison = pnl_results.comparison.copy()

    true_pnl = pnl_comparison["True PNL"]
    pca_pnl = pnl_comparison["PCA PNL"]
    pnl_diff = pnl_comparison["Diff"]
    pnl_diff_pct = pnl_comparison["Diff %"]

    mean_abs_true_pnl = true_pnl.abs().mean()
    mean_abs_diff = pnl_diff.abs().mean()
    rmse_diff = float(np.sqrt(np.mean(np.square(pnl_diff))))
    tracking_error = float(pnl_diff.std(ddof=1)) if len(pnl_diff) >= 2 else np.nan
    pnl_correlation = float(true_pnl.corr(pca_pnl)) if len(pnl_comparison) >= 2 else np.nan

    true_pnl_variance = float(true_pnl.var(ddof=1)) if len(true_pnl) >= 2 else np.nan
    diff_variance = float(pnl_diff.var(ddof=1)) if len(pnl_diff) >= 2 else np.nan
    hedge_effectiveness = np.nan
    if pd.notna(true_pnl_variance) and true_pnl_variance > 0 and pd.notna(diff_variance):
        hedge_effectiveness = 1.0 - diff_variance / true_pnl_variance

    sign_match_ratio = float((np.sign(true_pnl) == np.sign(pca_pnl)).mean())

    return {
        "mean_abs_true_pnl": float(mean_abs_true_pnl),
        "mean_abs_diff": float(mean_abs_diff),
        "mean_abs_diff_ratio": float(mean_abs_diff / mean_abs_true_pnl) if mean_abs_true_pnl != 0 else np.nan,
        "rmse_diff": rmse_diff,
        "tracking_error": tracking_error,
        "pnl_correlation": pnl_correlation,
        "hedge_effectiveness": float(hedge_effectiveness) if pd.notna(hedge_effectiveness) else np.nan,
        "mean_abs_diff_pct": float(pnl_diff_pct.abs().mean()),
        "median_abs_diff_pct": float(pnl_diff_pct.abs().median()),
        "p95_abs_diff_pct": float(pnl_diff_pct.abs().quantile(0.95)),
        "max_abs_diff_pct": float(pnl_diff_pct.abs().max()),
        "mean_signed_diff": float(pnl_diff.mean()),
        "sign_match_ratio": sign_match_ratio,
    }


def evaluate_single_static_configuration(
    raw_market_data: pd.DataFrame,
    config: DashboardConfig,
    preprocessing_spec: Dict,
    covariance_spec: Dict,
    n_components: int,
    candidate_hedge_sets: List[Tuple[str, ...]],
):
    processed_market_data, pca_model = fit_static_pca_model(
        raw_market_data=raw_market_data,
        config=config,
        preprocessing_spec=preprocessing_spec,
        covariance_spec=covariance_spec,
        n_components=n_components,
    )

    reconstructed_levels, actual_levels, residual_levels = pca_model.reconstruct(revert_processing=True)
    reconstruction_summary = reconstruction_metrics(actual_levels, residual_levels)

    model_level_metrics = {
        "explained_var_topk": float(np.sum(pca_model.results.explained_var_ratios[:n_components])),
        "rmse_global": float(reconstruction_summary["RMSE Global"]),
        "mae_global": float(reconstruction_summary["MAE Global"]),
        "r2": float(reconstruction_summary["R2"]),
    }

    per_hedge_rows = []
    per_hedge_artifacts = {}

    for hedge_tenors in candidate_hedge_sets:
        if len(hedge_tenors) != n_components:
            continue

        try:
            pnl_results = backtest_pca_hedge(
                pca=pca_model,
                true_sensi_position=config.true_sensitivity_vector,
                hedge_tenors=list(hedge_tenors),
                horizon_days=config.pnl_horizon_days,
            )
            pnl_summary = summarize_pnl_quality(pnl_results)
            loading_condition_number = compute_loading_condition_number(pca_model, hedge_tenors)
        except Exception as exception_message:
            per_hedge_rows.append({
                "preprocessing_label": preprocessing_label(preprocessing_spec),
                "covariance_label": covariance_label(covariance_spec),
                "n_components": n_components,
                "hedge_tenors": hedge_tenors,
                "error": str(exception_message),
            })
            continue

        row = {
            "preprocessing_label": preprocessing_label(preprocessing_spec),
            "covariance_label": covariance_label(covariance_spec),
            "n_components": n_components,
            "hedge_tenors": hedge_tenors,
            "loading_condition_number": loading_condition_number,
            **model_level_metrics,
            **pnl_summary,
        }
        per_hedge_rows.append(row)

        model_identifier = (
            preprocessing_label(preprocessing_spec),
            covariance_label(covariance_spec),
            n_components,
            hedge_tenors,
        )
        per_hedge_artifacts[model_identifier] = {
            "processed_market_data": processed_market_data,
            "pca_model": pca_model,
            "pnl_results": pnl_results,
            "reconstructed_levels": reconstructed_levels,
            "actual_levels": actual_levels,
            "residual_levels": residual_levels,
        }

    return per_hedge_rows, per_hedge_artifacts


def add_convenience_ranking(static_results: pd.DataFrame, ranking_weights: Dict[str, float]) -> pd.DataFrame:
    ranked_results = static_results.copy()

    valid_rows = ranked_results.dropna(subset=[
        "mean_abs_diff_ratio",
        "rmse_diff",
        "p95_abs_diff_pct",
        "pnl_correlation",
        "loading_condition_number",
        "explained_var_topk",
    ]).copy()

    if valid_rows.empty:
        ranked_results["convenience_score"] = np.nan
        return ranked_results

    valid_rows["pnl_correlation_penalty"] = 1.0 - valid_rows["pnl_correlation"].clip(-1.0, 1.0)
    valid_rows["explained_var_penalty"] = 1.0 - valid_rows["explained_var_topk"].clip(0.0, 1.0)

    score_components = [
        "mean_abs_diff_ratio",
        "rmse_diff",
        "p95_abs_diff_pct",
        "pnl_correlation_penalty",
        "loading_condition_number",
        "explained_var_penalty",
    ]

    for metric_name in score_components:
        metric_series = valid_rows[metric_name].replace([np.inf, -np.inf], np.nan)
        lower_quantile = metric_series.quantile(0.05)
        upper_quantile = metric_series.quantile(0.95)
        clipped_series = metric_series.clip(lower=lower_quantile, upper=upper_quantile)
        denominator = clipped_series.max() - clipped_series.min()
        if denominator == 0 or not np.isfinite(denominator):
            valid_rows[f"{metric_name}_normalized"] = 0.0
        else:
            valid_rows[f"{metric_name}_normalized"] = (clipped_series - clipped_series.min()) / denominator

    valid_rows["convenience_score"] = 0.0
    for metric_name, metric_weight in ranking_weights.items():
        valid_rows["convenience_score"] += metric_weight * valid_rows[f"{metric_name}_normalized"]

    ranked_results = ranked_results.merge(
        valid_rows[["preprocessing_label", "covariance_label", "n_components", "hedge_tenors", "convenience_score"]],
        on=["preprocessing_label", "covariance_label", "n_components", "hedge_tenors"],
        how="left",
    )

    return ranked_results


# --------------------------------------------------------------------------------------
# Rolling-study functions
# --------------------------------------------------------------------------------------
def component_similarity_matrix(reference_loadings: pd.DataFrame, candidate_loadings: pd.DataFrame) -> np.ndarray:
    reference_matrix = reference_loadings.values
    candidate_matrix = candidate_loadings.values

    reference_norms = np.linalg.norm(reference_matrix, axis=0, keepdims=True)
    candidate_norms = np.linalg.norm(candidate_matrix, axis=0, keepdims=True)

    reference_unit_vectors = reference_matrix / reference_norms
    candidate_unit_vectors = candidate_matrix / candidate_norms
    return reference_unit_vectors.T @ candidate_unit_vectors


def best_permutation_from_similarity(similarity_matrix: np.ndarray) -> Tuple[int, ...]:
    n_components = similarity_matrix.shape[0]
    best_permutation: Optional[Tuple[int, ...]] = None
    best_score = -np.inf
    for permutation in itertools.permutations(range(n_components)):
        permutation_score = sum(abs(similarity_matrix[row_index, permutation[row_index]]) for row_index in range(n_components))
        if permutation_score > best_score:
            best_score = permutation_score
            best_permutation = tuple(permutation)
    if best_permutation is None:
        raise ValueError("Unable to determine permutation for loading alignment.")
    return best_permutation


def align_loadings_to_reference(reference_loadings: pd.DataFrame, current_loadings: pd.DataFrame) -> pd.DataFrame:
    similarity_matrix = component_similarity_matrix(reference_loadings, current_loadings)
    best_permutation = best_permutation_from_similarity(similarity_matrix)
    aligned_loadings = current_loadings.iloc[:, list(best_permutation)].copy()
    aligned_loadings.columns = reference_loadings.columns

    for component_name in reference_loadings.columns:
        signed_dot_product = float(np.dot(reference_loadings[component_name].values, aligned_loadings[component_name].values))
        if signed_dot_product < 0:
            aligned_loadings[component_name] *= -1.0

    return aligned_loadings


def fit_pca_on_processed_window(
    processed_window: pd.DataFrame,
    raw_window: pd.DataFrame,
    config: DashboardConfig,
    covariance_spec: Dict,
    n_components: int,
) -> StaticPCA:
    temporary_market_data = clone_market_data_with_preprocessing(
        raw_market_data=raw_window,
        config=config,
        preprocessing_spec={"compute_changes": False, "demean": False, "standardize": False},
    )
    temporary_market_data.processed_data = processed_window.copy()
    temporary_market_data.data = raw_window.copy()

    covariance_estimator = build_covariance_estimator(covariance_spec)
    pca_model = StaticPCA(
        data=temporary_market_data,
        n_components=n_components,
        cov_estimator=covariance_estimator,
    )
    pca_model.fit()
    return pca_model


def run_rolling_loading_study(
    raw_market_data: pd.DataFrame,
    config: DashboardConfig,
    preprocessing_spec: Dict,
    covariance_spec: Dict,
    n_components: int,
    window_rows: int,
    step_rows: int,
) -> pd.DataFrame:
    processed_market_data = clone_market_data_with_preprocessing(raw_market_data, config, preprocessing_spec)
    processed_data = processed_market_data.processed_data.copy()
    raw_data = processed_market_data.data.copy().loc[processed_data.index.min():processed_data.index.max()]

    rolling_rows = []
    previous_loadings = None

    for end_row in range(window_rows, len(processed_data) + 1, step_rows):
        processed_window = processed_data.iloc[end_row - window_rows:end_row].copy()
        raw_window = raw_data.loc[processed_window.index.min():processed_window.index.max()].copy()

        pca_model = fit_pca_on_processed_window(
            processed_window=processed_window,
            raw_window=raw_window,
            config=config,
            covariance_spec=covariance_spec,
            n_components=n_components,
        )

        current_loadings = pca_model.results.loadings.copy()
        if previous_loadings is not None:
            current_loadings = align_loadings_to_reference(previous_loadings, current_loadings)

        rolling_row = {
            "window_end": processed_window.index[-1],
            "explained_var_topk": float(np.sum(pca_model.results.explained_var_ratios[:n_components])),
        }

        for component_index in range(n_components):
            rolling_row[f"PC{component_index + 1}_explained_var"] = float(pca_model.results.explained_var_ratios[component_index])
            if previous_loadings is not None:
                previous_vector = previous_loadings.iloc[:, component_index].values
                current_vector = current_loadings.iloc[:, component_index].values
                cosine_similarity = float(
                    np.dot(previous_vector, current_vector)
                    / (np.linalg.norm(previous_vector) * np.linalg.norm(current_vector))
                )
                rolling_row[f"PC{component_index + 1}_cosine_similarity"] = cosine_similarity
            else:
                rolling_row[f"PC{component_index + 1}_cosine_similarity"] = np.nan

        rolling_rows.append(rolling_row)
        previous_loadings = current_loadings.copy()

    if not rolling_rows:
        return pd.DataFrame()

    return pd.DataFrame(rolling_rows).set_index("window_end")


def run_rolling_out_of_sample_pnl_study(
    raw_market_data: pd.DataFrame,
    config: DashboardConfig,
    preprocessing_spec: Dict,
    covariance_spec: Dict,
    n_components: int,
    hedge_tenors: Tuple[str, ...],
    window_rows: int,
    step_rows: int,
) -> pd.DataFrame:
    processed_market_data = clone_market_data_with_preprocessing(raw_market_data, config, preprocessing_spec)
    processed_data = processed_market_data.processed_data.copy()
    raw_data = processed_market_data.data.copy().sort_index()

    rolling_pnl_blocks = []

    for end_row in range(window_rows, len(processed_data) - step_rows + 1, step_rows):
        processed_train = processed_data.iloc[end_row - window_rows:end_row].copy()
        raw_train = raw_data.loc[processed_train.index.min():processed_train.index.max()].copy()

        pca_model = fit_pca_on_processed_window(
            processed_window=processed_train,
            raw_window=raw_train,
            config=config,
            covariance_spec=covariance_spec,
            n_components=n_components,
        )

        projection_matrix = project_pca_loadings_on_hedge_tenors(
            loadings=pca_model.results.loadings,
            hedge_tenors=list(hedge_tenors),
        )
        pca_sensitivity = compute_pca_hedge(
            sensi=config.true_sensitivity_vector,
            proj_matrix=projection_matrix,
        )

        test_start_index = end_row
        test_end_index = min(end_row + step_rows + config.pnl_horizon_days, len(raw_data))
        raw_test_slice = raw_data.iloc[test_start_index:test_end_index].copy()
        if len(raw_test_slice) <= config.pnl_horizon_days:
            continue

        pnl_results = compare_true_and_pca_pnl(
            market_data=raw_test_slice,
            true_sensi_position=config.true_sensitivity_vector,
            pca_sensi=pca_sensitivity,
            horizon_days=config.pnl_horizon_days,
        )
        pnl_frame = pnl_results.comparison.copy().iloc[:step_rows].copy()
        pnl_frame["refit_date"] = processed_train.index[-1]
        rolling_pnl_blocks.append(pnl_frame)

    if not rolling_pnl_blocks:
        return pd.DataFrame()

    return pd.concat(rolling_pnl_blocks).sort_index()


def summarize_rolling_out_of_sample_pnl(rolling_pnl_frame: pd.DataFrame) -> pd.Series:
    if rolling_pnl_frame.empty:
        return pd.Series(dtype=float)

    true_pnl = rolling_pnl_frame["True PNL"]
    pca_pnl = rolling_pnl_frame["PCA PNL"]
    pnl_diff = rolling_pnl_frame["Diff"]
    pnl_diff_pct = rolling_pnl_frame["Diff %"]

    return pd.Series({
        "mean_abs_diff": float(pnl_diff.abs().mean()),
        "rmse_diff": float(np.sqrt(np.mean(np.square(pnl_diff)))),
        "mean_abs_diff_pct": float(pnl_diff_pct.abs().mean()),
        "p95_abs_diff_pct": float(pnl_diff_pct.abs().quantile(0.95)),
        "pnl_correlation": float(true_pnl.corr(pca_pnl)) if len(rolling_pnl_frame) >= 2 else np.nan,
        "sign_match_ratio": float((np.sign(true_pnl) == np.sign(pca_pnl)).mean()),
    })


def assign_market_regimes_from_price_moves(price_moves: pd.DataFrame) -> pd.Series:
    stress_indicator = price_moves.abs().mean(axis=1)
    return pd.qcut(
        stress_indicator.rank(method="first"),
        q=4,
        labels=["calm", "normal", "active", "stressed"],
    )


def regime_breakdown_table(pnl_results) -> pd.DataFrame:
    pnl_comparison = pnl_results.comparison.copy()
    regimes = assign_market_regimes_from_price_moves(pnl_results.price_moves)
    pnl_comparison["regime"] = regimes

    rows = []
    for regime_name, regime_frame in pnl_comparison.groupby("regime"):
        rows.append({
            "regime": regime_name,
            "n_obs": int(len(regime_frame)),
            "mean_abs_diff": float(regime_frame["Diff"].abs().mean()),
            "rmse_diff": float(np.sqrt(np.mean(np.square(regime_frame["Diff"])) )) if len(regime_frame) > 0 else np.nan,
            "mean_abs_diff_pct": float(regime_frame["Diff %"].abs().mean()),
            "pnl_corr": float(regime_frame["True PNL"].corr(regime_frame["PCA PNL"])) if len(regime_frame) >= 2 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("regime")


# --------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------
def make_curve_panel_figure(raw_market_data: pd.DataFrame, selected_tenors: List[str]):
    figure, axis = plt.subplots(figsize=(14, 5))
    for tenor_name in selected_tenors:
        if tenor_name in raw_market_data.columns:
            axis.plot(raw_market_data.index, raw_market_data[tenor_name], label=tenor_name)
    axis.set_title("Raw market history")
    axis.set_ylabel("Level")
    axis.legend(ncol=min(len(selected_tenors), 5))
    axis.grid(True, alpha=0.3)
    return figure


def make_daily_change_volatility_figure(raw_market_data: pd.DataFrame):
    daily_changes = raw_market_data.diff().dropna()
    annualized_volatility = daily_changes.std() * np.sqrt(252)

    figure, axis = plt.subplots(figsize=(14, 5))
    annualized_volatility.plot(kind="bar", ax=axis)
    axis.set_title("Annualized volatility of daily changes by tenor")
    axis.set_ylabel("Volatility")
    axis.grid(True, axis="y", alpha=0.3)
    return figure


def make_correlation_heatmap_figure(raw_market_data: pd.DataFrame, use_changes: bool = True):
    data_for_correlation = raw_market_data.diff().dropna() if use_changes else raw_market_data.copy()
    correlation_matrix = data_for_correlation.corr()

    figure, axis = plt.subplots(figsize=(10, 8))
    heatmap = axis.imshow(correlation_matrix.values, aspect="auto")
    axis.set_title("Correlation matrix" + (" of daily changes" if use_changes else " of levels"))
    axis.set_xticks(range(len(correlation_matrix.columns)))
    axis.set_xticklabels(correlation_matrix.columns, rotation=90)
    axis.set_yticks(range(len(correlation_matrix.index)))
    axis.set_yticklabels(correlation_matrix.index)
    figure.colorbar(heatmap, ax=axis)
    return figure


def make_metric_frontier_figure(static_results: pd.DataFrame):
    valid_results = static_results.dropna(subset=["explained_var_topk", "mean_abs_diff_ratio", "n_components"])
    figure, axis = plt.subplots(figsize=(10, 6))
    if valid_results.empty:
        axis.text(0.5, 0.5, "No valid results to plot", ha="center", va="center")
        axis.set_axis_off()
        return figure

    scatter = axis.scatter(
        valid_results["explained_var_topk"],
        valid_results["mean_abs_diff_ratio"],
        c=valid_results["n_components"],
        s=40,
        alpha=0.7,
    )
    axis.set_xlabel("Explained variance of retained PCs")
    axis.set_ylabel("Mean absolute hedge error / mean absolute true PNL")
    axis.set_title("Statistical fit versus hedge usefulness")
    axis.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axis, label="n_components")
    return figure


def make_loadings_figure(pca_model: StaticPCA, title: str = ""):
    loadings = pca_model.results.loadings.copy()
    figure, axis = plt.subplots(figsize=(14, 5))
    for component_name in loadings.columns:
        axis.plot(loadings.index, loadings[component_name], marker="o", label=component_name)
    axis.axhline(0.0, linestyle="--", linewidth=1.0)
    axis.set_title(f"PCA loadings {title}")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return figure


def make_explained_variance_figure(pca_model: StaticPCA, title: str = ""):
    explained_variance = pca_model.results.explained_var_ratios[:pca_model.n_components]
    cumulative_explained_variance = np.cumsum(explained_variance)
    component_labels = [f"PC{i + 1}" for i in range(len(explained_variance))]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(component_labels, explained_variance)
    axis.plot(component_labels, cumulative_explained_variance, marker="o")
    axis.set_title(f"Explained variance {title}")
    axis.set_ylabel("Share of variance")
    axis.grid(True, axis="y", alpha=0.3)
    return figure


def make_reconstruction_figure(actual_levels: pd.DataFrame, reconstructed_levels: pd.DataFrame, selected_tenors: List[str]):
    figure, axis = plt.subplots(figsize=(14, 5))
    for tenor_name in selected_tenors:
        if tenor_name not in actual_levels.columns:
            continue
        axis.plot(actual_levels.index, actual_levels[tenor_name], label=f"{tenor_name} actual")
        axis.plot(reconstructed_levels.index, reconstructed_levels[tenor_name], linestyle="--", label=f"{tenor_name} reconstructed")
    axis.set_title("Reconstruction on selected tenors")
    axis.legend(ncol=2)
    axis.grid(True, alpha=0.3)
    return figure


def make_pnl_diagnostics_figure(pnl_results, title: str = ""):
    pnl_comparison = pnl_results.comparison.copy()
    figure, axes = plt.subplots(3, 1, figsize=(14, 14))

    axes[0].plot(pnl_comparison.index, pnl_comparison["True PNL"], label="True PNL")
    axes[0].plot(pnl_comparison.index, pnl_comparison["PCA PNL"], label="PCA PNL")
    axes[0].set_title(f"True PNL versus PCA PNL {title}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pnl_comparison.index, pnl_comparison["Diff"])
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_title("Tracking error time series")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(pnl_comparison["True PNL"], pnl_comparison["PCA PNL"], alpha=0.5)
    minimum_value = np.nanmin([pnl_comparison["True PNL"].min(), pnl_comparison["PCA PNL"].min()])
    maximum_value = np.nanmax([pnl_comparison["True PNL"].max(), pnl_comparison["PCA PNL"].max()])
    axes[2].plot([minimum_value, maximum_value], [minimum_value, maximum_value], linestyle="--")
    axes[2].set_xlabel("True PNL")
    axes[2].set_ylabel("PCA PNL")
    axes[2].set_title("Scatter: hedge replication quality")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return figure


def make_rolling_loading_figure(rolling_loading_frame: pd.DataFrame, title: str = ""):
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    if rolling_loading_frame.empty:
        axes[0].text(0.5, 0.5, "No rolling loading results", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No rolling loading results", ha="center", va="center")
        return figure

    similarity_columns = [column_name for column_name in rolling_loading_frame.columns if column_name.endswith("cosine_similarity")]
    explained_variance_columns = [column_name for column_name in rolling_loading_frame.columns if column_name.endswith("explained_var") and column_name != "explained_var_topk"]

    if similarity_columns:
        rolling_loading_frame[similarity_columns].plot(ax=axes[0])
    axes[0].set_title(f"Rolling loading stability {title}")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].grid(True, alpha=0.3)

    if explained_variance_columns:
        rolling_loading_frame[explained_variance_columns + ["explained_var_topk"]].plot(ax=axes[1])
    axes[1].set_title("Explained variance through time")
    axes[1].set_ylabel("Variance share")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return figure


def make_rolling_pnl_figure(rolling_pnl_frame: pd.DataFrame, title: str = ""):
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    if rolling_pnl_frame.empty:
        axes[0].text(0.5, 0.5, "No rolling PNL results", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No rolling PNL results", ha="center", va="center")
        return figure

    axes[0].plot(rolling_pnl_frame.index, rolling_pnl_frame["True PNL"], label="True PNL")
    axes[0].plot(rolling_pnl_frame.index, rolling_pnl_frame["PCA PNL"], label="PCA PNL")
    axes[0].set_title(f"Rolling out-of-sample PNL replication {title}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rolling_pnl_frame.index, rolling_pnl_frame["Diff"])
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_title("Rolling out-of-sample tracking error")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return figure


# --------------------------------------------------------------------------------------
# Study runners
# --------------------------------------------------------------------------------------
def estimate_total_static_tests(config: DashboardConfig) -> int:
    total_configurations = 0
    for n_components in config.n_components_grid:
        add_all_combinations = n_components <= config.add_all_combinations_up_to_k
        candidate_hedge_sets = build_candidate_hedge_sets(
            n_components=n_components,
            hedge_universe=config.hedge_search_universe,
            manual_sets_dictionary=config.manual_hedge_sets,
            add_all_combinations=add_all_combinations,
            max_combinations=config.max_hedge_combinations_per_k,
        )
        total_configurations += len(candidate_hedge_sets)

    return total_configurations * len(config.preprocessing_grid) * len(config.covariance_grid) // max(len(config.n_components_grid), 1)


def run_static_study(raw_market_data: pd.DataFrame, config: DashboardConfig):
    static_rows = []
    static_artifacts = {}

    all_configuration_blocks = []
    for preprocessing_spec in config.preprocessing_grid:
        for covariance_spec in config.covariance_grid:
            for n_components in config.n_components_grid:
                add_all_combinations = n_components <= config.add_all_combinations_up_to_k
                candidate_hedge_sets = build_candidate_hedge_sets(
                    n_components=n_components,
                    hedge_universe=config.hedge_search_universe,
                    manual_sets_dictionary=config.manual_hedge_sets,
                    add_all_combinations=add_all_combinations,
                    max_combinations=config.max_hedge_combinations_per_k,
                )
                all_configuration_blocks.append((preprocessing_spec, covariance_spec, n_components, candidate_hedge_sets))

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    for configuration_index, (preprocessing_spec, covariance_spec, n_components, candidate_hedge_sets) in enumerate(all_configuration_blocks, start=1):
        status_placeholder.write(
            f"Running static study: {preprocessing_label(preprocessing_spec)} | {covariance_label(covariance_spec)} | "
            f"k={n_components} | hedge sets={len(candidate_hedge_sets)}"
        )
        current_rows, current_artifacts = evaluate_single_static_configuration(
            raw_market_data=raw_market_data,
            config=config,
            preprocessing_spec=preprocessing_spec,
            covariance_spec=covariance_spec,
            n_components=n_components,
            candidate_hedge_sets=candidate_hedge_sets,
        )
        static_rows.extend(current_rows)
        static_artifacts.update(current_artifacts)
        progress_bar.progress(configuration_index / len(all_configuration_blocks))

    progress_bar.empty()
    status_placeholder.empty()

    static_results = pd.DataFrame(static_rows)
    if static_results.empty:
        return static_results, static_artifacts

    static_results = add_convenience_ranking(static_results, config.ranking_weights)
    static_results["hedge_tenors_display"] = static_results["hedge_tenors"].apply(lambda hedge_tuple: ", ".join(hedge_tuple) if isinstance(hedge_tuple, tuple) else hedge_tuple)
    return static_results, static_artifacts


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
def render_kpi_row(kpi_values: Dict[str, float]):
    columns = st.columns(len(kpi_values))
    for column, (label, value) in zip(columns, kpi_values.items()):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            column.metric(label, "NA")
        else:
            column.metric(label, f"{value:,.4f}")


def get_selected_model_key(selection_index: int, ranked_results: pd.DataFrame):
    selected_row = ranked_results.iloc[selection_index]
    return (
        selected_row["preprocessing_label"],
        selected_row["covariance_label"],
        int(selected_row["n_components"]),
        tuple(selected_row["hedge_tenors"]),
    )


# --------------------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------------------
def main():
    st.title("PCA hedge model selection dashboard")
    st.caption("Interactive dashboard for PCA hedge diagnostics, PNL replication, and rolling stability analysis.")

    with st.sidebar:
        st.header("Inputs")

        uploaded_file = st.file_uploader("Upload historical curve data", type=["csv", "xlsx", "xls"])
        local_file_path = st.text_input("Or local file path", value="")

        curve_index = st.text_input("Curve index", value="ESTER")
        curve_currency = st.text_input("Currency", value="EUR")
        curve_identifier = st.text_input("Curve ID", value="YOUR_CURVE_ID")
        curve_closing = st.text_input("Closing", value="CLOSE")

        start_date = st.text_input("Start date (dd/mm/yy)", value="01/01/20")
        end_date = st.text_input("End date (dd/mm/yy)", value="31/12/25")

        pnl_horizon_days = st.number_input("PNL horizon days", min_value=1, max_value=60, value=5, step=1)

        st.subheader("Model grids")
        preprocessing_choices = st.multiselect(
            "Preprocessing choices",
            options=[
                "chg=1|demean=1|std=0",
                "chg=1|demean=0|std=1",
                "chg=0|demean=1|std=0",
                "chg=0|demean=0|std=1",
            ],
            default=DEFAULT_PREPROCESSING_CHOICES,
        )
        covariance_choices = st.multiselect(
            "Covariance estimators",
            options=["sample", "ledoit_wolf", "OAS", "ewma_hl_20", "ewma_hl_60", "graphical_lasso", "graphical_lasso_cv"],
            default=DEFAULT_COVARIANCE_CHOICES,
        )
        n_components_grid = st.multiselect("Numbers of components", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4])

        st.subheader("Hedge search")
        hedge_search_universe_input = st.text_input(
            "Hedge search universe (comma-separated)",
            value=", ".join(DEFAULT_HEDGE_SEARCH_UNIVERSE),
        )
        manual_hedge_sets_text = st.text_area(
            "Manual hedge sets; separate sets with ';' and tenors with ','",
            value="10Y; 15Y; 20Y; 5Y,10Y; 5Y,30Y; 10Y,30Y; 10Y,20Y; 5Y,10Y,30Y; 5Y,15Y,30Y; 2Y,10Y,30Y; 2Y,5Y,10Y,30Y",
            height=110,
        )
        add_all_combinations_up_to_k = st.number_input("Add all combinations up to k", min_value=1, max_value=5, value=3, step=1)
        max_hedge_combinations_per_k = st.number_input("Max hedge combinations per k", min_value=1, max_value=10000, value=2000, step=50)

        st.subheader("Rolling study")
        rolling_window_rows_values = st.multiselect("Rolling windows (rows)", options=[63, 126, 252, 504, 756], default=[126, 252, 504])
        rolling_step_rows = st.number_input("Rolling step rows", min_value=1, max_value=252, value=21, step=1)

    if uploaded_file is None and not local_file_path:
        st.info("Upload a CSV/XLSX file or provide a local file path to start the analysis.")
        st.stop()

    if uploaded_file is not None:
        source_path = save_uploaded_file(uploaded_file)
    else:
        source_path = local_file_path.strip()

    if not source_path:
        st.warning("No data source available.")
        st.stop()

    raw_preview = pd.read_csv(source_path, index_col=0) if source_path.lower().endswith(".csv") else pd.read_excel(source_path, index_col=0)
    raw_preview = ensure_datetime_index(raw_preview)
    available_columns = list(raw_preview.columns)

    with st.sidebar:
        st.subheader("Tenors")
        default_tenors = [tenor for tenor in DEFAULT_TENORS if tenor in available_columns]
        if not default_tenors:
            default_tenors = available_columns[: min(len(available_columns), 20)]
        rates_tenors_universe = st.multiselect("Curve tenor universe", options=available_columns, default=default_tenors)
        plot_sample_tenors = st.multiselect("Plot sample tenors", options=rates_tenors_universe, default=[tenor for tenor in DEFAULT_PLOT_TENORS if tenor in rates_tenors_universe] or rates_tenors_universe[:4])

    st.subheader("True sensitivity vector")
    if "sensitivity_editor" not in st.session_state or list(st.session_state["sensitivity_editor"]["tenor"]) != rates_tenors_universe:
        initial_sensitivity_dataframe = make_default_sensitivity_dataframe(rates_tenors_universe)
        if {"5Y", "10Y", "15Y", "20Y", "30Y"}.intersection(rates_tenors_universe):
            for tenor_name, sensitivity_value in {"5Y": 125000.0, "10Y": 210000.0, "15Y": -150000.0, "20Y": 90000.0, "30Y": -120000.0}.items():
                if tenor_name in initial_sensitivity_dataframe["tenor"].values:
                    initial_sensitivity_dataframe.loc[initial_sensitivity_dataframe["tenor"] == tenor_name, "sensitivity"] = sensitivity_value
        st.session_state["sensitivity_editor"] = initial_sensitivity_dataframe

    edited_sensitivity_dataframe = st.data_editor(
        st.session_state["sensitivity_editor"],
        num_rows="fixed",
        use_container_width=True,
        key="sensitivity_data_editor",
        column_config={
            "tenor": st.column_config.TextColumn("Tenor", disabled=True),
            "sensitivity": st.column_config.NumberColumn("Sensitivity", format="%.6f"),
        },
    )
    st.session_state["sensitivity_editor"] = edited_sensitivity_dataframe

    parsed_hedge_search_universe = [item.strip() for item in hedge_search_universe_input.split(",") if item.strip()]
    parsed_hedge_search_universe = [tenor for tenor in parsed_hedge_search_universe if tenor in rates_tenors_universe]
    parsed_manual_hedge_sets = parse_tuple_text(manual_hedge_sets_text)

    curve_info = CurveInfo(
        index=curve_index,
        ccy=curve_currency,
        curve_id=curve_identifier,
        closing=curve_closing,
        is_rate=True,
    )

    config = DashboardConfig(
        source_path=source_path,
        curve_info=curve_info,
        start_date=start_date,
        end_date=end_date,
        rates_tenors_universe=rates_tenors_universe,
        true_sensitivity_vector=dataframe_to_sensitivity_series(edited_sensitivity_dataframe),
        preprocessing_grid=[parse_preprocessing_choice(choice_label) for choice_label in preprocessing_choices],
        covariance_grid=[parse_covariance_choice(choice_label) for choice_label in covariance_choices],
        n_components_grid=sorted(n_components_grid),
        hedge_search_universe=parsed_hedge_search_universe,
        manual_hedge_sets=parsed_manual_hedge_sets,
        add_all_combinations_up_to_k=int(add_all_combinations_up_to_k),
        max_hedge_combinations_per_k=int(max_hedge_combinations_per_k),
        pnl_horizon_days=int(pnl_horizon_days),
        rolling_window_rows_grid=sorted(int(value) for value in rolling_window_rows_values),
        rolling_step_rows=int(rolling_step_rows),
        plot_sample_tenors=plot_sample_tenors,
        ranking_weights=DEFAULT_RANKING_WEIGHTS.copy(),
    )

    if not config.rates_tenors_universe:
        st.error("Select at least one tenor in the curve universe.")
        st.stop()

    if not config.preprocessing_grid or not config.covariance_grid or not config.n_components_grid:
        st.error("At least one preprocessing choice, covariance estimator, and component count must be selected.")
        st.stop()

    try:
        base_market_data = make_base_market_data(config)
        raw_market_data = ensure_datetime_index(base_market_data.data.copy())
    except Exception as exception_message:
        st.error(f"Failed to load market data: {exception_message}")
        st.stop()

    overview_tab, static_tab, deep_dive_tab, rolling_tab = st.tabs([
        "Data overview",
        "Static study",
        "Deep dive",
        "Rolling study",
    ])

    with overview_tab:
        st.markdown("### Data sanity checks")
        render_kpi_row({
            "Rows": float(raw_market_data.shape[0]),
            "Tenors": float(raw_market_data.shape[1]),
        })
        st.write(f"Date range: {raw_market_data.index.min().date()} to {raw_market_data.index.max().date()}")
        st.dataframe(raw_market_data.head(), use_container_width=True)

        chart_column_1, chart_column_2 = st.columns(2)
        with chart_column_1:
            st.pyplot(make_curve_panel_figure(raw_market_data, config.plot_sample_tenors))
        with chart_column_2:
            st.pyplot(make_daily_change_volatility_figure(raw_market_data))
        st.pyplot(make_correlation_heatmap_figure(raw_market_data, use_changes=True))

    with static_tab:
        st.markdown("### Static PCA study")
        total_grid_size = len(config.preprocessing_grid) * len(config.covariance_grid) * len(config.n_components_grid)
        st.write(f"Grid blocks to evaluate: {total_grid_size}")

        if st.button("Run static study", type="primary"):
            try:
                static_results, static_artifacts = run_static_study(raw_market_data, config)
                st.session_state["static_results"] = static_results
                st.session_state["static_artifacts"] = static_artifacts
            except Exception as exception_message:
                st.error(f"Static study failed: {exception_message}")

        if "static_results" in st.session_state and not st.session_state["static_results"].empty:
            static_results = st.session_state["static_results"].copy()
            static_artifacts = st.session_state.get("static_artifacts", {})

            clean_results = static_results.dropna(subset=["mean_abs_diff_ratio", "pnl_correlation"]).copy()
            ranked_results = clean_results.sort_values(
                ["convenience_score", "mean_abs_diff_ratio", "pnl_correlation"],
                ascending=[True, True, False],
            )
            st.session_state["ranked_results"] = ranked_results

            render_kpi_row({
                "Tested configurations": float(len(static_results)),
                "Valid configurations": float(len(ranked_results)),
                "Best score": float(ranked_results["convenience_score"].min()) if not ranked_results.empty else np.nan,
            })

            st.pyplot(make_metric_frontier_figure(ranked_results))

            results_to_display = ranked_results[[
                "preprocessing_label",
                "covariance_label",
                "n_components",
                "hedge_tenors_display",
                "explained_var_topk",
                "rmse_global",
                "r2",
                "mean_abs_diff_ratio",
                "rmse_diff",
                "p95_abs_diff_pct",
                "pnl_correlation",
                "hedge_effectiveness",
                "loading_condition_number",
                "convenience_score",
            ]].head(50)
            st.dataframe(results_to_display, use_container_width=True)

            csv_bytes = ranked_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download ranked static results",
                data=csv_bytes,
                file_name="pca_static_results.csv",
                mime="text/csv",
            )

            summary_columns = st.columns(3)
            with summary_columns[0]:
                st.markdown("#### Median metrics by components")
                st.dataframe(
                    ranked_results.groupby("n_components")[["explained_var_topk", "mean_abs_diff_ratio", "pnl_correlation", "loading_condition_number"]].median(),
                    use_container_width=True,
                )
            with summary_columns[1]:
                st.markdown("#### Median metrics by preprocessing")
                st.dataframe(
                    ranked_results.groupby("preprocessing_label")[["explained_var_topk", "mean_abs_diff_ratio", "pnl_correlation", "loading_condition_number"]].median().sort_values("mean_abs_diff_ratio"),
                    use_container_width=True,
                )
            with summary_columns[2]:
                st.markdown("#### Median metrics by covariance")
                st.dataframe(
                    ranked_results.groupby("covariance_label")[["explained_var_topk", "mean_abs_diff_ratio", "pnl_correlation", "loading_condition_number"]].median().sort_values("mean_abs_diff_ratio"),
                    use_container_width=True,
                )
        elif "static_results" in st.session_state:
            st.warning("No valid static results were produced.")

    with deep_dive_tab:
        st.markdown("### Deep dive on shortlisted models")
        ranked_results = st.session_state.get("ranked_results")
        static_artifacts = st.session_state.get("static_artifacts", {})

        if ranked_results is None or ranked_results.empty:
            st.info("Run the static study first.")
        else:
            shortlist_size = st.slider("Number of shortlisted models", min_value=1, max_value=min(20, len(ranked_results)), value=min(5, len(ranked_results)))
            shortlist = ranked_results.head(shortlist_size).copy()
            shortlist_display = shortlist[[
                "preprocessing_label",
                "covariance_label",
                "n_components",
                "hedge_tenors_display",
                "explained_var_topk",
                "mean_abs_diff_ratio",
                "pnl_correlation",
                "hedge_effectiveness",
                "convenience_score",
            ]]
            st.dataframe(shortlist_display, use_container_width=True)

            selected_shortlist_index = st.selectbox(
                "Model to inspect",
                options=list(range(len(shortlist))),
                format_func=lambda row_number: (
                    f"#{row_number + 1} | {shortlist.iloc[row_number]['preprocessing_label']} | "
                    f"{shortlist.iloc[row_number]['covariance_label']} | "
                    f"k={int(shortlist.iloc[row_number]['n_components'])} | "
                    f"hedge={shortlist.iloc[row_number]['hedge_tenors_display']}"
                ),
            )

            selected_key = (
                shortlist.iloc[selected_shortlist_index]["preprocessing_label"],
                shortlist.iloc[selected_shortlist_index]["covariance_label"],
                int(shortlist.iloc[selected_shortlist_index]["n_components"]),
                tuple(shortlist.iloc[selected_shortlist_index]["hedge_tenors"]),
            )
            selected_artifact = static_artifacts[selected_key]
            selected_pca_model = selected_artifact["pca_model"]
            selected_pnl_results = selected_artifact["pnl_results"]

            render_kpi_row({
                "Explained var": float(shortlist.iloc[selected_shortlist_index]["explained_var_topk"]),
                "Mean abs diff ratio": float(shortlist.iloc[selected_shortlist_index]["mean_abs_diff_ratio"]),
                "PNL correlation": float(shortlist.iloc[selected_shortlist_index]["pnl_correlation"]),
                "Hedge effectiveness": float(shortlist.iloc[selected_shortlist_index]["hedge_effectiveness"]),
            })

            chart_column_1, chart_column_2 = st.columns(2)
            with chart_column_1:
                st.pyplot(make_explained_variance_figure(selected_pca_model))
            with chart_column_2:
                st.pyplot(make_loadings_figure(selected_pca_model))

            st.pyplot(make_reconstruction_figure(
                actual_levels=selected_artifact["actual_levels"],
                reconstructed_levels=selected_artifact["reconstructed_levels"],
                selected_tenors=config.plot_sample_tenors,
            ))
            st.pyplot(make_pnl_diagnostics_figure(selected_pnl_results))

            detail_column_1, detail_column_2 = st.columns(2)
            with detail_column_1:
                st.markdown("#### PCA hedge sensitivities")
                st.dataframe(selected_pnl_results.pca_sensi.to_frame("pca_sensi"), use_container_width=True)
                st.markdown("#### Regime breakdown")
                st.dataframe(regime_breakdown_table(selected_pnl_results), use_container_width=True)
            with detail_column_2:
                st.markdown("#### PNL comparison")
                st.dataframe(selected_pnl_results.comparison.head(50), use_container_width=True)
                st.markdown("#### Projection matrix")
                st.dataframe(selected_pnl_results.projection_matrix, use_container_width=True)

    with rolling_tab:
        st.markdown("### Rolling stability and out-of-sample hedge study")
        ranked_results = st.session_state.get("ranked_results")
        static_artifacts = st.session_state.get("static_artifacts", {})

        if ranked_results is None or ranked_results.empty:
            st.info("Run the static study first to create a shortlist.")
        else:
            max_shortlist_for_rolling = min(10, len(ranked_results))
            rolling_shortlist_size = st.slider("Shortlist size for rolling study", min_value=1, max_value=max_shortlist_for_rolling, value=min(3, max_shortlist_for_rolling))
            rolling_shortlist = ranked_results.head(rolling_shortlist_size).copy()
            st.dataframe(
                rolling_shortlist[["preprocessing_label", "covariance_label", "n_components", "hedge_tenors_display", "mean_abs_diff_ratio", "pnl_correlation", "convenience_score"]],
                use_container_width=True,
            )

            if st.button("Run rolling study"):
                rolling_summary_rows = []
                rolling_detailed_results = {}
                total_rolling_runs = len(rolling_shortlist) * max(len(config.rolling_window_rows_grid), 1)
                progress_bar = st.progress(0.0)
                run_counter = 0

                try:
                    for _, candidate_row in rolling_shortlist.iterrows():
                        preprocessing_spec = next(
                            spec for spec in config.preprocessing_grid if preprocessing_label(spec) == candidate_row["preprocessing_label"]
                        )
                        covariance_spec = next(
                            spec for spec in config.covariance_grid if covariance_label(spec) == candidate_row["covariance_label"]
                        )
                        n_components = int(candidate_row["n_components"])
                        hedge_tenors = tuple(candidate_row["hedge_tenors"])

                        for window_rows in config.rolling_window_rows_grid:
                            rolling_loading_frame = run_rolling_loading_study(
                                raw_market_data=raw_market_data,
                                config=config,
                                preprocessing_spec=preprocessing_spec,
                                covariance_spec=covariance_spec,
                                n_components=n_components,
                                window_rows=window_rows,
                                step_rows=config.rolling_step_rows,
                            )
                            rolling_pnl_frame = run_rolling_out_of_sample_pnl_study(
                                raw_market_data=raw_market_data,
                                config=config,
                                preprocessing_spec=preprocessing_spec,
                                covariance_spec=covariance_spec,
                                n_components=n_components,
                                hedge_tenors=hedge_tenors,
                                window_rows=window_rows,
                                step_rows=config.rolling_step_rows,
                            )
                            rolling_pnl_summary = summarize_rolling_out_of_sample_pnl(rolling_pnl_frame)
                            mean_loading_similarity = np.nanmean([
                                rolling_loading_frame[column_name].mean()
                                for column_name in rolling_loading_frame.columns
                                if column_name.endswith("cosine_similarity")
                            ]) if not rolling_loading_frame.empty else np.nan

                            summary_row = {
                                "preprocessing_label": candidate_row["preprocessing_label"],
                                "covariance_label": candidate_row["covariance_label"],
                                "n_components": n_components,
                                "hedge_tenors": hedge_tenors,
                                "window_rows": window_rows,
                                "mean_loading_similarity": float(mean_loading_similarity) if pd.notna(mean_loading_similarity) else np.nan,
                                **rolling_pnl_summary.to_dict(),
                            }
                            rolling_summary_rows.append(summary_row)

                            rolling_key = (
                                candidate_row["preprocessing_label"],
                                candidate_row["covariance_label"],
                                n_components,
                                hedge_tenors,
                                window_rows,
                            )
                            rolling_detailed_results[rolling_key] = {
                                "rolling_loading_frame": rolling_loading_frame,
                                "rolling_pnl_frame": rolling_pnl_frame,
                            }

                            run_counter += 1
                            progress_bar.progress(run_counter / total_rolling_runs)

                    rolling_summary = pd.DataFrame(rolling_summary_rows).sort_values(
                        ["rmse_diff", "mean_abs_diff_pct", "mean_loading_similarity"],
                        ascending=[True, True, False],
                    )
                    rolling_summary["hedge_tenors_display"] = rolling_summary["hedge_tenors"].apply(lambda hedge_tuple: ", ".join(hedge_tuple))
                    st.session_state["rolling_summary"] = rolling_summary
                    st.session_state["rolling_detailed_results"] = rolling_detailed_results
                    progress_bar.empty()
                except Exception as exception_message:
                    st.error(f"Rolling study failed: {exception_message}")

            rolling_summary = st.session_state.get("rolling_summary")
            rolling_detailed_results = st.session_state.get("rolling_detailed_results", {})

            if rolling_summary is not None and not rolling_summary.empty:
                st.dataframe(
                    rolling_summary[[
                        "preprocessing_label",
                        "covariance_label",
                        "n_components",
                        "hedge_tenors_display",
                        "window_rows",
                        "mean_loading_similarity",
                        "mean_abs_diff",
                        "rmse_diff",
                        "mean_abs_diff_pct",
                        "p95_abs_diff_pct",
                        "pnl_correlation",
                        "sign_match_ratio",
                    ]],
                    use_container_width=True,
                )

                selected_rolling_index = st.selectbox(
                    "Rolling configuration to inspect",
                    options=list(range(len(rolling_summary))),
                    format_func=lambda row_number: (
                        f"#{row_number + 1} | {rolling_summary.iloc[row_number]['preprocessing_label']} | "
                        f"{rolling_summary.iloc[row_number]['covariance_label']} | "
                        f"k={int(rolling_summary.iloc[row_number]['n_components'])} | "
                        f"hedge={rolling_summary.iloc[row_number]['hedge_tenors_display']} | "
                        f"window={int(rolling_summary.iloc[row_number]['window_rows'])}"
                    ),
                )

                selected_rolling_row = rolling_summary.iloc[selected_rolling_index]
                rolling_key = (
                    selected_rolling_row["preprocessing_label"],
                    selected_rolling_row["covariance_label"],
                    int(selected_rolling_row["n_components"]),
                    tuple(selected_rolling_row["hedge_tenors"]),
                    int(selected_rolling_row["window_rows"]),
                )
                selected_payload = rolling_detailed_results[rolling_key]
                st.pyplot(make_rolling_loading_figure(selected_payload["rolling_loading_frame"], title=str(rolling_key)))
                st.pyplot(make_rolling_pnl_figure(selected_payload["rolling_pnl_frame"], title=str(rolling_key)))
                st.dataframe(selected_payload["rolling_pnl_frame"].head(100), use_container_width=True)


if __name__ == "__main__":
    main()
