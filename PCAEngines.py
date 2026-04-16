import numpy as np
import pandas as pd

class RollingSVDPCA:
    def __init__(self, n_components=2, window=252, min_periods=None):
        self.n_components = n_components
        self.window = window
        self.min_periods = min_periods or window

    # ------------------------------------------------------------------
    # Core: fit one window via SVD
    # ------------------------------------------------------------------
    def _fit_window(self, X):
        """X: (window, p) centered data. Returns components (k, p), eigenvalues (k,), singular_values (k,)."""
        n = X.shape[0]
        mean = X.mean(axis=0)
        Xc = X - mean

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        k = min(self.n_components, S.shape[0])
        return {
            "mean":               mean,
            "components":         Vt[:k],                          # (k, p)
            "singular_values":    S[:k],
            "explained_variance": (S[:k] ** 2) / (n - 1),
            "explained_variance_ratio": (S[:k] ** 2) / (S ** 2).sum(),
            "loadings":           Vt[:k].T,                        # (p, k) unscaled
            "loadings_scaled":    Vt[:k].T * np.sqrt((S[:k] ** 2) / (n - 1)),  # (p, k)
        }

    # ------------------------------------------------------------------
    # Sign alignment: match current components to previous window
    # ------------------------------------------------------------------
    @staticmethod
    def _align_signs(current_components, prev_components):
        if prev_components is None:
            return current_components
        aligned = current_components.copy()
        for i in range(current_components.shape[0]):
            if np.dot(current_components[i], prev_components[i]) < 0:
                aligned[i] *= -1
        return aligned

    # ------------------------------------------------------------------
    # Main fit: roll over the full DataFrame
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """
        df: (T, p) DataFrame of returns (or any stationary series).
        Populates self.results_ — a list of dicts keyed by date.
        """
        data   = df.values
        dates  = df.index
        T, p   = data.shape
        k      = self.n_components

        # Storage
        self.results_ = []

        prev_components = None

        for end in range(self.min_periods, T + 1):
            start   = max(0, end - self.window)
            window  = data[start:end]

            if window.shape[0] < self.min_periods:
                continue

            date = dates[end - 1]
            res  = self._fit_window(window)

            # Align signs before storing
            res["components"] = self._align_signs(res["components"], prev_components)
            # Re-derive loadings after potential sign flip
            S = res["singular_values"]
            n = window.shape[0]
            res["loadings"]        = res["components"].T
            res["loadings_scaled"] = res["components"].T * np.sqrt(res["explained_variance"])

            prev_components = res["components"].copy()

            # Scores for the last observation in the window
            last_obs = window[-1] - res["mean"]
            res["scores_last"] = last_obs @ res["components"].T   # (k,)
            res["date"] = date

            self.results_.append(res)

        return self

    # ------------------------------------------------------------------
    # Convenience extractors → tidy DataFrames
    # ------------------------------------------------------------------
    def get_explained_variance_ratio(self) -> pd.DataFrame:
        dates = [r["date"] for r in self.results_]
        data  = np.array([r["explained_variance_ratio"] for r in self.results_])
        cols  = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(data, index=dates, columns=cols)

    def get_loadings(self, scaled=True) -> dict:
        """Returns {date: DataFrame(features × PCs)} for every window."""
        key  = "loadings_scaled" if scaled else "loadings"
        cols = [f"PC{i+1}" for i in range(self.n_components)]
        return {
            r["date"]: pd.DataFrame(r[key], columns=cols)
            for r in self.results_
        }

    def get_scores(self) -> pd.DataFrame:
        """Last-observation score for each rolling window."""
        dates = [r["date"] for r in self.results_]
        data  = np.array([r["scores_last"] for r in self.results_])
        cols  = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(data, index=dates, columns=cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Project df onto the components fitted at each date.
        Useful for out-of-sample projection using the most recent window's components.
        """
        if not self.results_:
            raise RuntimeError("Call fit() first.")
        last  = self.results_[-1]
        Xc    = df.values - last["mean"]
        proj  = Xc @ last["components"].T
        cols  = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(proj, index=df.index, columns=cols)
