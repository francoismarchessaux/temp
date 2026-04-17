import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from utils import get_business_days, excel_to_date
from pyJade.pearl import pearl_service


@dataclass
class WinsorizeReport:
    """
    Diagnostic report produced by _winsorize.

    Attributes
    ----------
    n_clipped_cells : int
        Total number of (date, column) cells that were clipped.
    n_clipped_dates : int
        Number of distinct dates on which at least one clipping occurred.
    clipped_cells : pd.DataFrame  (bool, same shape as input)
        True wherever a value was clipped.
    clip_magnitude : pd.DataFrame  (float, same shape as input)
        Absolute z-score of the original value at each clipped cell, 0 elsewhere.
        Useful for ranking which outliers were most extreme.
    """
    n_clipped_cells: int
    n_clipped_dates: int
    clipped_cells: pd.DataFrame
    clip_magnitude: pd.DataFrame

    def summary(self) -> str:
        return (
            f"Winsorization summary: {self.n_clipped_cells} cell(s) clipped "
            f"across {self.n_clipped_dates} date(s)."
        )


class DataLoader:
    def get_vol_at_date(self, index: str, ccy: str, surface: str, closing: str, date: pd.Timestamp) -> pd.Series:
        """ Retrieve vol surface at specified date and reshape it as a 2D vector """

        # Get vol surface object
        vol_curve_obj = pearl_service().ARM_GET_VALUE(
            pearl_service().ARM_OXYGEN_CreateIRVol(index, ccy, surface, closing, date), 0
        )

        # Get individual data
        vols = pearl_service().ARM_GET_VALUE(vol_curve_obj, 5)
        expiries = np.asarray(pearl_service().ARM_GET_VALUE(vol_curve_obj, 6)).flatten()
        tenors = np.asarray(pearl_service().ARM_GET_VALUE(vol_curve_obj, 7)).flatten()

        # Map vol to expiries/tenors
        vol_as_surface = pd.DataFrame(index=expiries, columns=tenors, data=vols)

        # Flatten surface as 1D vector
        vol_as_vec = vol_as_surface.stack()
        vol_as_vec.index = [f"{expiry}{tenor}" for expiry, tenor in vol_as_vec.index]
        return vol_as_vec

    @staticmethod
    def _winsorize(data: pd.DataFrame, z_threshold: float) -> tuple[pd.DataFrame, WinsorizeReport]:
        """
        Winsorize a vol surface DataFrame column-by-column using z-scores.

        For each column (= one expiry-tenor point), values whose z-score exceeds
        z_threshold in absolute value are clipped to mean ± z_threshold * std.

        Statistics (mean, std) are computed over the full column so they are
        consistent across the entire loaded date range. This is intentional:
        outlier detection should be calibrated globally, not locally, to avoid
        masking genuine extreme events as "local normal".

        Parameters
        ----------
        data        : (T x N) DataFrame of vol surfaces, dates as index
        z_threshold : clip anything with |z| > z_threshold (typical: 3.0 – 4.0)

        Returns
        -------
        clipped     : winsorized DataFrame, same shape and dtype
        report      : WinsorizeReport with diagnostic info
        """
        col_mean = data.mean(axis=0)
        col_std  = data.std(axis=0, ddof=1)

        # Avoid division by zero for constant columns (flat smile points, etc.)
        col_std_safe = col_std.replace(0.0, np.nan)

        z_scores = (data - col_mean) / col_std_safe

        lower = col_mean - z_threshold * col_std
        upper = col_mean + z_threshold * col_std

        clipped = data.clip(lower=lower, upper=upper, axis=1)

        # Diagnostic: which cells were actually moved
        clipped_mask = data != clipped
        clip_magnitude = z_scores.abs().where(clipped_mask, other=0.0)

        report = WinsorizeReport(
            n_clipped_cells=int(clipped_mask.values.sum()),
            n_clipped_dates=int(clipped_mask.any(axis=1).sum()),
            clipped_cells=clipped_mask,
            clip_magnitude=clip_magnitude,
        )
        return clipped, report

    def get_vol(
        self,
        index: str,
        ccy: str,
        surface: str,
        closing: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        winsorize: bool = False,
        z_threshold: float = 3.0,
    ) -> tuple[pd.DataFrame, WinsorizeReport | None]:
        """
        Retrieve vol surfaces for a range of dates.

        Parameters
        ----------
        winsorize   : if True, apply per-column z-score winsorization after loading
        z_threshold : number of standard deviations beyond which values are clipped.
                      Only used when winsorize=True. Typical range: 3.0 – 4.0.

        Returns
        -------
        data   : (T x N) DataFrame of (optionally winsorized) vol surfaces
        report : WinsorizeReport if winsorize=True, else None
        """
        # Get business dates
        xldates = get_business_days(start_date, end_date, ccy)
        dates = [excel_to_date(date) for date in xldates]

        # Fetch all data
        with ThreadPoolExecutor(max_workers=16) as executor:
            res = [executor.submit(self.get_vol_at_date, index, ccy, surface, closing, d) for d in dates]
            data_results = [x.result() for x in res]

        # Build DataFrame
        tenors = data_results[0].index
        data = pd.DataFrame(data=data_results, index=dates, columns=tenors).sort_index(ascending=True)

        # Preprocessing: winsorize
        if winsorize:
            data, report = self._winsorize(data, z_threshold)
            return data, report

        return data, None
