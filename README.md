# opto-fp-tools-open

Tools for processing optogenetic and fiber photometry (FP) data, with a focus on generating fMRI-ready regressors and QA outputs.

## What this repository is for

This repository provides utilities to:

- inspect `.doric`/HDF5 file contents,
- convert HDF5/`.doric` data into standardized parquet tables,
- split continuous FP recordings into run segments using TTL pulses,
- resample FP channels to BOLD/fMRI temporal resolution,
- optionally convolve signals with canonical HRFs (Glover/SPM),
- apply censor masks,
- generate regressor/contrast text files for GLM analyses,
- generate QA plots at multiple processing steps,
- run group-level FLAMEO maps from merged cope files.

## Repository layout

```text
.
├── README.md
├── extract_regressor.py              # End-to-end regressor extraction workflow
├── h5_tree.py                        # CLI utility to print HDF5 tree structure
├── hdf5_plotter.py                   # GUI viewer/plotter for HDF5 datasets
├── flameo_group_map.py               # Group-level FLAMEO helper
├── fp_processing/
│   └── hdf5_to_parquet.py            # HDF5/.doric -> standardized parquet conversion
├── bold_fp_eval/
│   ├── fp_runsplitter.py             # Detect run boundaries from TTL pulses
│   ├── fp_dividebyruns.py            # Slice data by run windows and time channels
│   ├── match_fp_to_bold.py           # Low-pass + downsample to BOLD TR grid
│   ├── fp_hemodynamicize.py          # HRF convolution (Glover or SPM)
│   ├── fp_censor.py                  # Apply censor mask to matched FP data
│   ├── stim_regressor.py             # Build regressor + contrast arrays
│   └── plot_regressor.py             # Plot generated regressor text files
└── QA/
    └── qa_fp_plots.py                # QA plotting helpers for pipeline checkpoints
```

## Requirements

This repository does not currently include a pinned environment file, so install dependencies manually.

### Python dependencies (from imports)

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `h5py`
- `pyarrow`
- `nilearn`

`tkinter` is required for `hdf5_plotter.py` (usually included with standard Python installs, depending on your OS build).

### External tools

- **FSL** (specifically `fslmerge` and `flameo`) for `flameo_group_map.py`

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib h5py pyarrow nilearn
```

## Core processing workflow

The high-level regressor generation pipeline implemented in `extract_regressor.py` is:

1. Convert raw `.doric` to standardized parquet (`hdf5_to_parquet`).
2. Detect run boundaries from TTL pulses (`fp_runsplitter`).
3. Split full recording into per-run parquet files (`fp_dividebyruns`).
4. Resample FP traces to BOLD timing (`match_fp_to_bold`).
5. Optionally convolve with HRF (`fp_hemodynamicize`).
6. Optionally censor volumes (`fp_censor`).
7. Build and save regressor/contrast text files (`stim_regressor`).
8. Optionally plot regressors (`plot_regressor`).

## Usage examples

### 1) Print HDF5/Doric tree

```bash
python /home/runner/work/opto-fp-tools-open/opto-fp-tools-open/h5_tree.py /absolute/path/to/file.doric
```

### 2) Interactive HDF5 viewer

```bash
python /home/runner/work/opto-fp-tools-open/opto-fp-tools-open/hdf5_plotter.py
```

### 3) Run the regressor extraction workflow from Python

```python
from extract_regressor import extract_regressor

extract_regressor(
    input_file="/absolute/path/to/input.doric",
    tr=1.2,
    num_vols=500,
    output_file="/absolute/path/to/output_regressor.txt",
    hrf="glover",        # "glover", "spm", or None
    censor_file=None,    # optional CSV mask path
    plot_reg=False
)
```

### 4) Run group-level FLAMEO helper

```python
from flameo_group_map import merged_file, group_map

merged = merged_file(
    mydir="/absolute/path/to/cope_dir",
    mergedpath="/absolute/path/to/all_merged_cope.nii.gz"
)

group_map(
    merged_cope_file=merged,
    group_mask_file="/absolute/path/to/group_mask.nii.gz",
    design_mat_file="/absolute/path/to/design.mat",
    contrast_file="/absolute/path/to/contrasts.con",
    cs_file="/absolute/path/to/covsplit.mat",
    output_dir="/absolute/path/to/flameo_output",
    runmode="flame1"
)
```

## Inputs and outputs at a glance

- **Primary raw input:** `.doric`/HDF5 FP files with TTL/opto channels.
- **Intermediate outputs:** standardized parquet files (`*.standardized.parquet`) and QA plots (`.png`).
- **Final analysis outputs:** GLM regressor text files (`*_regressor.txt`) and contrast files (`*_con.txt`).

## Notes and practical guidance

- The code expects consistent channel naming and matching time-channel mappings.
- Several scripts contain local, environment-specific example paths in `if __name__ == "__main__"` blocks; update to your paths before running.
- Validate run split detection and matching quality using QA plots before downstream GLM analysis.

## License

This project is licensed under the terms of the `LICENSE` file in this repository.
