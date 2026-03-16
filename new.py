

@dataclass
class HedgeProjectionResults:
    projection_matrix: pd.DataFrame
    hedge_tenor_loadings: pd.DataFrame
    fitted_loadings: pd.DataFrame
    loading_residuals: pd.DataFrame
    fit_metrics: pd.DataFrame
    regression_method: str
    ridge_alpha: float

@dataclass
class WalkForwardBacktestResults:
    pnl_timeseries: pd.DataFrame
    hedge_notionals: pd.DataFrame
    rebalance_summary: pd.DataFrame
    projection_matrices: Dict[pd.Timestamp, pd.DataFrame]
    projection_fit_metrics: Dict[pd.Timestamp, pd.DataFrame]
    metrics: Dict[str, float]
    
###################
##### HEDGING #####
###################

def project_pca_loadings_on_hedge_tenors(
    loadings: pd.DataFrame,
    hedge_tenors: List[str],
    regression_method: Literal["ols", "ridge"] = "ols",
    ridge_alpha: float = 0.0,
    return_details: bool = False
) -> Union[pd.DataFrame, HedgeProjectionResults]:
    """
    Project the full PCA loading matrix onto a reduced hedge-tenor universe.

    The objective is to find a projection matrix ``W`` such that:

        W @ X ~= L

    where
        - ``L`` is the full loading matrix of shape (N_tenors, K),
        - ``X`` is the sub-matrix of hedge-tenor loadings of shape (H, K),
        - ``W`` is the desired projection matrix of shape (N_tenors, H).

    For a full sensitivity vector ``s`` aligned with ``loadings.index``, the hedge notionals are

        hedge = s.T @ W

    which produces one hedge notional per hedge tenor.

    Parameters
    ----------
    loadings
        PCA loading matrix with rows indexed by the full tenor universe and columns indexed by PCs.
    hedge_tenors
        Tenors to retain as hedge pillars.
    regression_method
        "ols" uses the Moore-Penrose pseudo-inverse and supports H != K.
        "ridge" solves the regularized least-squares problem
            min_W ||W X - L||_F^2 + ridge_alpha ||W||_F^2.
    ridge_alpha
        Ridge penalty used only when ``regression_method='ridge'``.
    return_details
        If True, return diagnostics together with the projection matrix.

    Returns
    -------
    pd.DataFrame or HedgeProjectionResults
        Projection matrix only by default, or a richer diagnostics object when
        ``return_details=True``.
    """

    if loadings is None or loadings.empty:
        raise ValueError("Loadings matrix is empty. Fit the PCA first.")

    if len(hedge_tenors) == 0:
        raise ValueError("hedge_tenors must contain at least one tenor.")

    missing_tenors = [tenor for tenor in hedge_tenors if tenor not in loadings.index]
    if missing_tenors:
        raise ValueError(f"Hedge tenors not found in loading index: {missing_tenors}")

    full_loading_matrix = loadings.loc[:, :].astype(float).values
    hedge_tenor_loading_matrix = loadings.loc[hedge_tenors, :].astype(float).values

    if regression_method == "ols":
        regression_operator = np.linalg.pinv(hedge_tenor_loading_matrix)
    elif regression_method == "ridge":
        if ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative.")

        n_hedge_tenors = hedge_tenor_loading_matrix.shape[0]
        regularization_matrix = ridge_alpha * np.eye(n_hedge_tenors)
        regression_operator = (
            hedge_tenor_loading_matrix.T
            @ np.linalg.pinv(
                hedge_tenor_loading_matrix @ hedge_tenor_loading_matrix.T + regularization_matrix
            )
        )
    else:
        raise ValueError("regression_method must be either 'ols' or 'ridge'.")

    projection_matrix_values = full_loading_matrix @ regression_operator

    projection_matrix = pd.DataFrame(
        projection_matrix_values,
        index=loadings.index,
        columns=hedge_tenors
    )

    if not return_details:
        return projection_matrix

    fitted_loading_values = projection_matrix_values @ hedge_tenor_loading_matrix
    loading_residual_values = full_loading_matrix - fitted_loading_values

    loading_norms = np.linalg.norm(full_loading_matrix, axis=1)
    residual_norms = np.linalg.norm(loading_residual_values, axis=1)
    stable_denominator = np.where(loading_norms > 1.0e-12, loading_norms, 1.0)
    rowwise_r_squared = 1.0 - np.sum(loading_residual_values ** 2, axis=1) / np.where(
        np.sum(full_loading_matrix ** 2, axis=1) > 1.0e-12,
        np.sum(full_loading_matrix ** 2, axis=1),
        1.0
    )

    fit_metrics = pd.DataFrame({
        "loading_norm": loading_norms,
        "residual_norm": residual_norms,
        "relative_residual_norm": residual_norms / stable_denominator,
        "r_squared_no_intercept": rowwise_r_squared
    }, index=loadings.index)

    return HedgeProjectionResults(
        projection_matrix=projection_matrix,
        hedge_tenor_loadings=loadings.loc[hedge_tenors].copy(),
        fitted_loadings=pd.DataFrame(
            fitted_loading_values,
            index=loadings.index,
            columns=loadings.columns
        ),
        loading_residuals=pd.DataFrame(
            loading_residual_values,
            index=loadings.index,
            columns=loadings.columns
        ),
        fit_metrics=fit_metrics,
        regression_method=regression_method,
        ridge_alpha=ridge_alpha
    )


def compute_pca_hedge(
    sensi: pd.Series,
    proj_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Project a full-tenor sensitivity vector onto hedge pillars.

    sensi: Series indexed by full tenor universe.
    proj_matrix: DataFrame (N_tenors x H), index = same tenors, columns = hedge pillars.

    Returns: Series indexed by hedge pillars (hedge notionals).
    """

    sensi_aligned = sensi.reindex(proj_matrix.index).fillna(0.0)
    hedge = sensi_aligned.values @ proj_matrix.values

    return pd.Series(hedge, index=proj_matrix.columns)

def _period_to_offset(period: str):
    """Convert strings such as 10B, 5D, 2W, 3M, 1Y into pandas offsets."""

    if not isinstance(period, str) or len(period) < 2:
        raise ValueError(f"Unsupported period format: {period}")

    period = period.strip().upper()
    period_value = int(period[:-1])
    period_unit = period[-1]

    if period_value <= 0:
        raise ValueError(f"Period value must be positive: {period}")

    if period_unit == "B":
        return pd.offsets.BDay(period_value)
    if period_unit == "D":
        return pd.DateOffset(days=period_value)
    if period_unit == "W":
        return pd.DateOffset(weeks=period_value)
    if period_unit == "M":
        return pd.DateOffset(months=period_value)
    if period_unit == "Y":
        return pd.DateOffset(years=period_value)

    raise ValueError(f"Unsupported period unit in '{period}'. Use one of B, D, W, M, Y.")


def _first_index_on_or_after(date_index: pd.DatetimeIndex, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    position = date_index.searchsorted(target_date, side="left")
    if position >= len(date_index):
        return None
    return date_index[position]


def _last_index_on_or_before(date_index: pd.DatetimeIndex, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    position = date_index.searchsorted(target_date, side="right") - 1
    if position < 0:
        return None
    return date_index[position]


def _build_window_market_data(base_market_data: MarketData, window_raw_data: pd.DataFrame) -> MarketData:
    window_start = pd.Timestamp(window_raw_data.index.min())
    window_end = pd.Timestamp(window_raw_data.index.max())

    window_market_data = MarketData(
        curve_info=base_market_data.curve_info,
        source=base_market_data.source,
        start_date=window_start.strftime("%d/%m/%y"),
        end_date=window_end.strftime("%d/%m/%y"),
        rates_tenors_universe=base_market_data.rates_tenors_universe,
        vol_tenors_universe=base_market_data.vol_tenors_universe,
        vol_expiries_universe=base_market_data.vol_expiries_universe
    )

    window_market_data.data = window_raw_data.copy()
    window_market_data.preprocess(
        compute_changes=base_market_data.compute_changes,
        demean=base_market_data.demean,
        standardize=base_market_data.standardize
    )

    return window_market_data


def _get_sensitivity_vector_asof(
    sensitivities: Union[pd.DataFrame, pd.Series],
    rebalance_date: pd.Timestamp,
    target_columns: List[str]
) -> pd.Series:
    if isinstance(sensitivities, pd.Series):
        sensitivity_vector = sensitivities.reindex(target_columns).fillna(0.0).astype(float)
        sensitivity_vector.name = rebalance_date
        return sensitivity_vector

    if not isinstance(sensitivities.index, pd.DatetimeIndex):
        try:
            sensitivities = sensitivities.copy()
            sensitivities.index = pd.to_datetime(sensitivities.index)
        except Exception as exception:
            raise ValueError(f"Sensitivity history index must be date-like: {exception}")

    available_sensitivities = sensitivities.sort_index().loc[:rebalance_date]

    if available_sensitivities.empty:
        raise ValueError(f"No sensitivity vector available on or before {rebalance_date}.")

    sensitivity_vector = available_sensitivities.iloc[-1].reindex(target_columns).fillna(0.0).astype(float)
    sensitivity_vector.name = available_sensitivities.index[-1]
    return sensitivity_vector


def _compute_hedge_effectiveness(book_pnl: pd.Series, residual_pnl: pd.Series) -> float:
    book_variance = float(book_pnl.var(ddof=1))
    residual_variance = float(residual_pnl.var(ddof=1))

    if not np.isfinite(book_variance) or book_variance <= 0.0:
        return np.nan

    return 1.0 - residual_variance / book_variance




def walk_forward_pca_backtest(
    market_data: MarketData,
    sensitivities: Union[pd.DataFrame, pd.Series],
    hedge_tenors: List[str],
    n_components: int,
    cov_estimator: CovarianceEstimator,
    estimation_window: str = "6M",
    rebalance_frequency: str = "1M",
    regression_method: Literal["ols", "ridge"] = "ols",
    ridge_alpha: float = 0.0
) -> WalkForwardBacktestResults:
    """
    Run a true walk-forward PCA hedge backtest.

    For each rebalance date ``t``:
        1. fit the PCA on the trailing estimation window ending at ``t``;
        2. regress the full loading matrix onto the chosen hedge tenors;
        3. compute hedge notionals from the book sensitivity known at ``t``;
        4. hold the hedge constant until the next rebalance date and score realized PnL.

    Parameters
    ----------
    market_data
        Loaded and preprocessed ``MarketData`` instance.
    sensitivities
        Either a constant sensitivity ``Series`` indexed by tenor, or a historical
        ``DataFrame`` indexed by rebalance dates and column-aligned with the tenor universe.
    hedge_tenors
        Tenors retained as hedge pillars.
    n_components
        Number of principal components to retain in each estimation window.
    cov_estimator
        Covariance estimator used inside each static PCA fit.
    estimation_window
        Trailing window used to estimate PCA, for example ``6M`` or ``1Y``.
    rebalance_frequency
        Out-of-sample holding period, for example ``1M`` or ``5B``.
    regression_method
        ``ols`` or ``ridge`` for the hedge projection regression.
    ridge_alpha
        Ridge penalty used only with ``regression_method='ridge'``.
    """

    if market_data.data is None:
        raise ValueError("market_data.data is empty. Call load() first.")

    if market_data.processed_data is None:
        raise ValueError("market_data.processed_data is empty. Call preprocess() first.")

    raw_market_data = market_data.data.copy()

    if not isinstance(raw_market_data.index, pd.DatetimeIndex):
        raw_market_data.index = pd.to_datetime(raw_market_data.index)

    raw_market_data = raw_market_data.sort_index()

    tenor_universe = list(market_data.processed_data.columns)
    raw_market_data = raw_market_data.loc[:, tenor_universe]

    missing_hedge_tenors = [tenor for tenor in hedge_tenors if tenor not in tenor_universe]
    if missing_hedge_tenors:
        raise ValueError(f"Hedge tenors not found in market data universe: {missing_hedge_tenors}")

    estimation_offset = _period_to_offset(estimation_window)
    rebalance_offset = _period_to_offset(rebalance_frequency)

    all_dates = raw_market_data.index
    first_rebalance_target = all_dates[0] + estimation_offset
    rebalance_date = _first_index_on_or_after(all_dates, first_rebalance_target)

    if rebalance_date is None:
        raise ValueError("Not enough history to form the first estimation window.")

    pnl_path_list = []
    hedge_notional_history = []
    rebalance_summary_list = []
    projection_matrices = {}
    projection_fit_metrics = {}

    market_moves = raw_market_data.diff()

    while rebalance_date is not None:
        next_rebalance_target = rebalance_date + rebalance_offset
        next_rebalance_date = _first_index_on_or_after(all_dates, next_rebalance_target)

        if next_rebalance_date is None:
            break

        estimation_start_target = rebalance_date - estimation_offset
        estimation_start_date = _first_index_on_or_after(all_dates, estimation_start_target)

        if estimation_start_date is None:
            estimation_start_date = all_dates[0]

        estimation_window_raw_data = raw_market_data.loc[estimation_start_date:rebalance_date].copy()

        if estimation_window_raw_data.shape[0] < max(3, n_components + 1):
            rebalance_date = next_rebalance_date
            continue

        window_market_data = _build_window_market_data(market_data, estimation_window_raw_data)

        static_pca = StaticPCA(
            data=window_market_data,
            n_components=n_components,
            cov_estimator=cov_estimator
        )
        static_pca.fit()

        projection_results = project_pca_loadings_on_hedge_tenors(
            loadings=static_pca.results.loadings,
            hedge_tenors=hedge_tenors,
            regression_method=regression_method,
            ridge_alpha=ridge_alpha,
            return_details=True
        )

        sensitivity_vector = _get_sensitivity_vector_asof(
            sensitivities=sensitivities,
            rebalance_date=rebalance_date,
            target_columns=tenor_universe
        )

        hedge_notionals = compute_pca_hedge(
            sensi=sensitivity_vector,
            proj_matrix=projection_results.projection_matrix
        )

        evaluation_moves = market_moves.loc[(market_moves.index > rebalance_date) & (market_moves.index <= next_rebalance_date), tenor_universe]

        if evaluation_moves.empty:
            rebalance_date = next_rebalance_date
            continue

        book_pnl = evaluation_moves.mul(sensitivity_vector, axis=1).sum(axis=1)
        hedge_pnl = evaluation_moves.loc[:, hedge_tenors].mul(hedge_notionals, axis=1).sum(axis=1)
        residual_pnl = book_pnl - hedge_pnl

        rebalance_label = pd.Timestamp(rebalance_date)

        pnl_path_list.append(pd.DataFrame({
            "rebalance_date": rebalance_label,
            "book_pnl": book_pnl,
            "hedge_pnl": hedge_pnl,
            "residual_pnl": residual_pnl
        }, index=evaluation_moves.index))

        hedge_notional_history.append(pd.Series(hedge_notionals, name=rebalance_label))

        rebalance_summary_list.append({
            "rebalance_date": rebalance_label,
            "estimation_start_date": estimation_start_date,
            "estimation_end_date": rebalance_date,
            "next_rebalance_date": next_rebalance_date,
            "n_estimation_observations": int(estimation_window_raw_data.shape[0]),
            "n_evaluation_observations": int(evaluation_moves.shape[0]),
            "book_pnl_sum": float(book_pnl.sum()),
            "hedge_pnl_sum": float(hedge_pnl.sum()),
            "residual_pnl_sum": float(residual_pnl.sum()),
            "book_pnl_std": float(book_pnl.std(ddof=1)),
            "residual_pnl_std": float(residual_pnl.std(ddof=1)),
            "hedge_effectiveness": _compute_hedge_effectiveness(book_pnl, residual_pnl),
            "mean_projection_r_squared": float(projection_results.fit_metrics["r_squared_no_intercept"].mean())
        })

        projection_matrices[rebalance_label] = projection_results.projection_matrix
        projection_fit_metrics[rebalance_label] = projection_results.fit_metrics

        rebalance_date = next_rebalance_date

    if not pnl_path_list:
        raise ValueError("The backtest produced no out-of-sample evaluation windows. Check the date range and frequencies.")

    pnl_timeseries = pd.concat(pnl_path_list).sort_index()
    hedge_notionals = pd.DataFrame(hedge_notional_history).sort_index()
    rebalance_summary = pd.DataFrame(rebalance_summary_list).set_index("rebalance_date").sort_index()

    aggregate_metrics = {
        "n_rebalances": float(rebalance_summary.shape[0]),
        "total_book_pnl": float(pnl_timeseries["book_pnl"].sum()),
        "total_hedge_pnl": float(pnl_timeseries["hedge_pnl"].sum()),
        "total_residual_pnl": float(pnl_timeseries["residual_pnl"].sum()),
        "book_pnl_std": float(pnl_timeseries["book_pnl"].std(ddof=1)),
        "hedge_pnl_std": float(pnl_timeseries["hedge_pnl"].std(ddof=1)),
        "residual_pnl_std": float(pnl_timeseries["residual_pnl"].std(ddof=1)),
        "residual_pnl_rmse": float(np.sqrt(np.mean(np.square(pnl_timeseries["residual_pnl"])))),
        "residual_pnl_mae": float(np.mean(np.abs(pnl_timeseries["residual_pnl"]))),
        "hedge_effectiveness": _compute_hedge_effectiveness(
            pnl_timeseries["book_pnl"],
            pnl_timeseries["residual_pnl"]
        ),
        "correlation_book_vs_hedge": float(pnl_timeseries["book_pnl"].corr(pnl_timeseries["hedge_pnl"])),
        "mean_projection_r_squared": float(
            np.mean([fit_metrics["r_squared_no_intercept"].mean() for fit_metrics in projection_fit_metrics.values()])
        )
    }

    return WalkForwardBacktestResults(
        pnl_timeseries=pnl_timeseries,
        hedge_notionals=hedge_notionals,
        rebalance_summary=rebalance_summary,
        projection_matrices=projection_matrices,
        projection_fit_metrics=projection_fit_metrics,
        metrics=aggregate_metrics
    )
