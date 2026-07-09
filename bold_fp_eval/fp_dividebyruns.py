import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

def fp_dividebyruns(
    df: pd.DataFrame,
    run_splits_times: List[Tuple[float, float]],
    corresponding_time_cols: Dict[str, Union[str, List[str]]],
    *,
    drop_all_nan_rows: bool = True
) -> List[pd.DataFrame]:
    """
    Divide the DataFrame into multiple DataFrames based on run split times, where
    each signal column is sliced using its OWN reference time column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing several signal columns and several time columns.
    run_splits_times : list of float
        Times (in the same units as the time columns) indicating the END of each run.
        Run i spans [ start_time_i , end_time_i ] where start_time_0 = 0 and
        start_time_i = run_splits_times[i-1] for i > 0.
    corresponding_time_cols : dict
        Mapping: signal_column -> reference_time_column (or list where [0] is the time col).
        Columns not in this mapping are treated as time columns and are sliced using themselves.
    drop_all_nan_rows : bool, default True
        If True, drop rows that are NaN across all columns in the split (tightens the frame).

    Returns
    -------
    List[pd.DataFrame]
        One DataFrame per run. Each column is sliced using the start/end indices that come
        from its corresponding time column.

    Notes
    -----
    - Efficient: caches (start_idx, end_idx) for each distinct time column across all runs.
    - Robust: falls back to argmin(|t - times|) if the time column isn't monotonic.
    - Special case: 'ttl_TR' uses 'ttl_time' if present (non-NaN), else 'minicube_time'.
      If both present and non-NaN, raises ValueError (same intent as your error log).
    """

    # ---------- helpers ----------
    def _normalize_time_key(val: Union[str, List[str]]) -> str:
        if isinstance(val, (list, tuple)):
            return val[0]
        return val

    def _resolve_ref_time_col(col: str) -> Optional[str]:
        # Special case: ttl_TR uses whichever time column is populated
        if col == "ttl_TR":
            has_ttl = ("ttl_time" in df.columns) and (not df["ttl_time"].isna().all())
            has_mini = ("minicube_time" in df.columns) and (not df["minicube_time"].isna().all())
            if has_ttl and not has_mini:
                return "ttl_time"
            if has_mini and not has_ttl:
                return "minicube_time"
            if has_ttl and has_mini:
                raise ValueError(
                    'Both "ttl_time" and "minicube_time" contain data; ambiguous for "ttl_TR".'
                )
            return None  # neither is usable

        # If explicit mapping given, use it; otherwise if col is a time column itself, use itself
        if col in corresponding_time_cols:
            return _normalize_time_key(corresponding_time_cols[col])
        # treat as time column if present in df
        return col if col in df.columns else None

    def _nearest_indices_for_times(
        time_values: np.ndarray,
        time_index: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        For a monotonic ascending 'time_values', find nearest indices (into 'time_index') for each target.
        If not monotonic, falls back to O(n*m) argmin per target.
        Returns indices into the ORIGINAL df index (via time_index).
        """
        if len(time_values) == 0:
            return np.array([None] * len(targets), dtype=object)

        # Check monotonic non-decreasing (allow small float jitters)
        monotonic = np.all(np.diff(time_values) >= 0)

        if monotonic:
            # binary search with nearest neighbor decision
            pos = np.searchsorted(time_values, targets, side="left")  # shape (m,)
            pos_clamped = np.clip(pos, 0, len(time_values) - 1)

            # compute candidates on left (pos-1) and right (pos), compare distances
            left = np.clip(pos - 1, 0, len(time_values) - 1)
            right = pos_clamped

            left_dist = np.abs(time_values[left] - targets)
            right_dist = np.abs(time_values[right] - targets)
            choose_left = left_dist <= right_dist
            chosen = np.where(choose_left, left, right)

            return time_index[chosen]
        else:
            # fallback: brute force nearest for each target
            out = []
            for t in targets:
                j = int(np.nanargmin(np.abs(time_values - t)))  # valid_values guaranteed non-empty
                out.append(time_index[j])
            return np.array(out, dtype=int)

    # ---------- precompute: which time column each df column uses ----------
    col_to_timecol: Dict[str, Optional[str]] = {}
    distinct_time_cols: set = set()
    for col in df.columns:
        ref = _resolve_ref_time_col(col)
        col_to_timecol[col] = ref
        if ref is not None:
            distinct_time_cols.add(ref)

    # Remove unusable time columns (all NaN or missing)
    valid_time_cols = {}
    for tcol in distinct_time_cols:
        if (tcol in df.columns) and (not df[tcol].isna().all()):
            valid_time_cols[tcol] = df[tcol]
        # else: we silently skip; columns depending on it will be skipped later

    # ---------- cache start/end indices per run per time column ----------
    # For run i: start_time = 0 (or previous split), end_time = run_splits_times[i]
    run_count = len(run_splits_times)
    start_times = np.array([t[0] for t in run_splits_times], dtype=float)
    end_times = np.array([t[1] for t in run_splits_times], dtype=float)

    # Map: time_col -> np.ndarray of shape (run_count, 2) with (start_idx, end_idx)
    timecol_to_bounds: Dict[str, np.ndarray] = {}

    for tcol, s in valid_time_cols.items():
        mask = ~s.isna().to_numpy()
        if not mask.any():
            continue
        time_vals = s.to_numpy()[mask]
        # original integer positions for the non-NaN time rows
        time_pos = np.flatnonzero(mask).astype(int)

        # Find nearest indices in ORIGINAL df index space
        start_idx_arr = _nearest_indices_for_times(time_vals, time_pos, start_times)
        end_idx_arr = _nearest_indices_for_times(time_vals, time_pos, end_times)

        # Ensure ints and start<=end (swap if needed)
        bounds = np.column_stack([start_idx_arr, end_idx_arr]).astype(int, copy=False)
        # Guarantee ascending (end exclusive slice later), if equal it's an empty slice
        swap_mask = bounds[:, 1] < bounds[:, 0]
        if np.any(swap_mask):
            bounds[swap_mask] = bounds[swap_mask][:, ::-1]
        timecol_to_bounds[tcol] = bounds

    # ---------- build each run split DataFrame ----------
    run_splits_df_list: List[pd.DataFrame] = []

    for i in range(run_count):
        col_slices = {}

        for col in df.columns:
            # If the whole column is NaN, skip early
            s = df[col]
            if s.isna().all():
                continue

            tcol = col_to_timecol.get(col)
            # if we don't have a usable time column for this signal/time column, skip it
            if tcol is None or tcol not in timecol_to_bounds:
                continue

            start_idx, end_idx = map(int, timecol_to_bounds[tcol][i])
            # end is exclusive, mirror original .iloc[start:end] behavior
            if start_idx == end_idx:
                continue  # empty slice, skip

            col_slices[col] = s.iloc[start_idx:end_idx]

        if not col_slices:
            # No columns could be sliced for this run; append empty DataFrame
            run_df = pd.DataFrame()
        else:
            # Let pandas align on the UNION of indices across slices (memory efficient)
            run_df = pd.DataFrame(col_slices)
            if drop_all_nan_rows:
                run_df = run_df.dropna(how="all")

        run_splits_df_list.append(run_df)

    return run_splits_df_list


if __name__ == "__main__":
    # Example usage 
    file = '/Users/samlaxer/Documents/Projects/ds-NcModfMRIOpto/derivatives/g2_withControl/procd/sub-45Ear436913/ses-20251122/fp/sub-45Ear436913_ses-20251122_desc-OptoScanRuns1and2_acq-0004_fp_norm_bandpass-01to1.standardized.parquet'
    # Get run split times from external source or define manually
    import fp_runsplitter
    df = pd.read_parquet(file)
    # num_vols like either 400 (9.4 T protocol) or 500 (15.2 T protocol)
    num_vols = 500
    # TR likely either 1.5 (9.4 T protocol) or 1.2 (15.2 T protocol)
    TR = 1.2
    run_split_times = fp_runsplitter.fp_runsplitter(df, TR, num_vols, output_plot=None)
    corresponding_time_cols = {
            "400-410nm_roi1": ['400-410nm_time'],
            "400-410nm_roi2": ['400-410nm_time'],
            "400-410nm_roi3": ['400-410nm_time'],
            "460-490nm_roi1": ['460-490nm_time'],
            "460-490nm_roi2": ['460-490nm_time'],
            "460-490nm_roi3": ['460-490nm_time'],
            "555-570nm_roi1": ['555-570nm_time'],
            "555-570nm_roi2": ['555-570nm_time'],
            "555-570nm_roi3": ['555-570nm_time'],
            "minicube_405nm": ['minicube_time'],
            "minicube_465nm": ['minicube_time'],
            "ttl_TR": ['ttl_time'],
            "ttl_stim": ['ttl_time'],
            "opto_DIO1": ['ttl_time'],
            "opto_DIO2": ['ttl_time'],
            "460-490nm_roi1_norm": ['time_norm'],
            "460-490nm_roi2_norm": ['time_norm'],
            "460-490nm_roi3_norm": ['time_norm'],
            "555-570nm_roi1_norm": ['time_norm'],
            "555-570nm_roi2_norm": ['time_norm'],
            "555-570nm_roi3_norm": ['time_norm'],
            "minicube_465nm_norm": ['time_norm']
        }

    result = fp_dividebyruns(
        df,
        run_splits_times=run_split_times,  # Example times
        corresponding_time_cols=corresponding_time_cols
    )

    print(result)




