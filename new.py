def explained_variance_over_time(self) -> pd.DataFrame:
    """
    Return explained variance ratios per rolling window.
    """
    if not self.pcas_results:
        raise ValueError("Run fit() first.")

    rows = []
    for i, pca in enumerate(self.pcas_results):
        end_date = pca.results.scores.index[-1]

        row = {"window_end": end_date}
        for k in range(self.n_components):
            row[f"PC{k+1}"] = float(pca.results.explained_var_ratios[k])

        row["TopK"] = float(np.sum(pca.results.explained_var_ratios[:self.n_components]))
        rows.append(row)

    return pd.DataFrame(rows).set_index("window_end")

def loading_stability(self) -> pd.DataFrame:
    """
    Cosine similarity of aligned loadings between consecutive windows.
    Assumes align_components() has already been run.
    """
    if len(self.pcas_results) < 2:
        raise ValueError("Need at least two rolling windows.")

    rows = []

    for i in range(1, len(self.pcas_results)):
        prev_pca = self.pcas_results[i - 1]
        cur_pca = self.pcas_results[i]

        row = {
            "window_end": cur_pca.results.scores.index[-1]
        }

        for k in range(self.n_components):
            prev_v = prev_pca.results.loadings.iloc[:, k].values
            cur_v = cur_pca.results.loadings.iloc[:, k].values

            sim = np.dot(prev_v, cur_v) / (
                np.linalg.norm(prev_v) * np.linalg.norm(cur_v)
            )
            row[f"PC{k+1}_similarity_to_prev"] = float(sim)

        rows.append(row)

    return pd.DataFrame(rows).set_index("window_end")

# similarity near 1.0 → factor is very stable
# similarity around 0.7–0.8 → factor is drifting
# similarity much lower → genuine regime break or noisy estimation


def _parse_surface_node(self, label: str):
    """
    Parse a concatenated node label like '3M10Y' or '18M5Y'
    using the MarketData universes.
    """
    expiries = sorted(self.data.vol_expiries_universe, key=len, reverse=True)
    tenors = sorted(self.data.vol_tenors_universe, key=len, reverse=True)

    expiry = next((e for e in expiries if label.startswith(e)), None)
    tenor = None

    if expiry is not None:
        suffix = label[len(expiry):]
        tenor = next((t for t in tenors if suffix == t), None)

    if expiry is None or tenor is None:
        raise ValueError(f"Cannot parse node label '{label}'")

    return expiry, tenor


def _to_years(self, x: str) -> float:
    if x.endswith("M"):
        return float(x[:-1]) / 12.0
    if x.endswith("Y"):
        return float(x[:-1])
    raise ValueError(f"Unknown tenor format '{x}'")

def _node_zone(self, label: str) -> str:
    expiry, tenor = self._parse_surface_node(label)
    expiry_y = self._to_years(expiry)
    tenor_y = self._to_years(tenor)

    expiry_bucket = "short_expiry" if expiry_y <= 2.0 else "long_expiry"
    tenor_bucket = "short_tenor" if tenor_y <= 5.0 else "long_tenor"

    return f"{expiry_bucket}|{tenor_bucket}"

def loading_zone_concentration(self) -> pd.DataFrame:
    """
    Share of absolute loading mass by zone for each PC and window.
    """
    if not self.pcas_results:
        raise ValueError("Run fit() first.")

    rows = []

    for pca in self.pcas_results:
        window_end = pca.results.scores.index[-1]
        loadings = pca.results.loadings.copy()

        zones = pd.Series(loadings.index, index=loadings.index).map(self._node_zone)

        for pc in loadings.columns:
            abs_load = loadings[pc].abs()
            total = abs_load.sum()

            row = {
                "window_end": window_end,
                "pc": pc
            }

            for zone in sorted(zones.unique()):
                zone_mass = abs_load[zones == zone].sum()
                row[zone] = float(zone_mass / total) if total > 0 else np.nan

            rows.append(row)

    return pd.DataFrame(rows)

def loading_zone_signs(self) -> pd.DataFrame:
    """
    Average signed loading by zone for each PC and window.
    """
    if not self.pcas_results:
        raise ValueError("Run fit() first.")

    rows = []

    for pca in self.pcas_results:
        window_end = pca.results.scores.index[-1]
        loadings = pca.results.loadings.copy()

        zones = pd.Series(loadings.index, index=loadings.index).map(self._node_zone)

        for pc in loadings.columns:
            row = {
                "window_end": window_end,
                "pc": pc
            }

            for zone in sorted(zones.unique()):
                row[zone] = float(loadings.loc[zones == zone, pc].mean())

            rows.append(row)

    return pd.DataFrame(rows)

# Now you can produce statements like:
# - PC2 is concentrated in short_expiry|short_tenor
# - the sign there is negative
# - so under a positive PC2 score, upper-left tends to move down relative to the rest
