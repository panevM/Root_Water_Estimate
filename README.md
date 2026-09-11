# Root Zone Water Estimation with Satellite Data

A speculative root-zone water estimator for students studying probabilistic robotics.
The teaching crop is **assumed spring wheat with zero assumed irrigation**. The configured field
near Skopje is unverified; neither crop nor management is known. No field-measured
root-zone observations exist. Agreement with an NDMI heuristic cannot establish field accuracy.

## Current delivery status

The default cache contains **90 genuine Open-Meteo ERA5 daily weather records and 45
Sentinel-2 daily satellite records**, covering 2025-05-01 through 2025-07-29, UTC.
After quality filtering, 26 NDMI observations are accepted and 19 are retained as rejected
records with their reasons. No measurements or timestamps are invented. The notebook uses
this saved dataset for reproducible offline runs; the root-zone estimate remains speculative
because no field-measured soil-water ground truth is available.

## Run

Use CPython 3.12 (standard Windows Python, rather than MSYS Python).
From this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-tested.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m root_zone_water.runner
.\.venv\Scripts\python execute_notebook.py
```

On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python` instead.
Open `Root_Zone_Water_Estimation_with_Satellite_Data.ipynb` in a Jupyter-capable editor and select this environment
for interactive Run All. `execute_notebook.py` executes a fresh kernel and replaces the
notebook's saved outputs. It also blocks outbound network connects inside the kernel;
local Jupyter communication remains available. Cached mode never loads `.env`.
This implementation was tested using the local `.venv-win` environment; that environment
is ignored and is not required by portable source paths.

`config.json` controls the crop, soil, assumed calendar, area, dates, prior, Q/R assumptions,
quality policy, Kc mode, and selected filter. `root_zone_water.runner.run()` honors the configured
filter; the notebook compares all three filter instances. Prediction outputs go to `outputs/`.
The notebook figures are interactive Plotly charts: use the wheel or drag to zoom, the pan tool
to move, double-click to reset, and legend clicks to hide or show traces. Each filter has its own
figure and is exported as `outputs/open_loop-state.html`,
`outputs/ekf-state.html`, and `outputs/ukf-state.html`, with Plotly embedded locally for offline use.

## Contents

- `Root_Zone_Water_Estimation_with_Satellite_Data.ipynb`: equations, agriculture, provenance and executable comparisons.
- `Root_Zone_Water_Estimation_with_Satellite_Data_mk.ipynb`: Macedonian translation of the same teaching notebook and calculations.
- `Create_Custom_Dataset.ipynb`: generate a personal config, download a new dataset, and connect it to offline estimation.
- `root_zone_water/model.py`, `proxy.py`: explicit daily water budget, active root depth and shared proxy.
- `root_zone_water/filters/`: minimal scalar filter interface, course covariance update and UKF.
- `root_zone_water/acquisition.py`, `data.py`: API acquisition and offline replay of raw responses.
- `root_zone_water/runner.py`, `plotting.py`: chronological processing, diagnostics, figures and sensitivity.
- `references/`: separate sourced crop/soil selections, with assumptions distinguished.
- `docs/API_SETUP.md`: exact remaining satellite setup and capture commands.
- `docs/DATASET.md`: schemas, provenance, timing, quality and cache contracts.
- `docs/EXTENDING_FILTERS.md`: filter extension guidance.

Open either notebook in Jupyter. To regenerate them from their authoring scripts, run
`python build_notebook.py` for English or `python build_notebook_mk.py` for Macedonian.
Both use the same `root_zone_water` modules and saved dataset; the Macedonian HTML figures
use `-mk` filenames so they do not overwrite the English exports.

The open-loop model, EKF and UKF use identical forcing policies. Each accepted acquisition
is corrected once at its assigned day-end boundary. Newly observed NDVI can only affect the
following interval. This retrospective calculation uses complete-day weather, not real-time forecasts.

## Create a personal dataset

Open `Create_Custom_Dataset.ipynb`. Edit `START_DATE`, `END_DATE`, `FILTER_NAME`,
`DATASET_PATH` and `CONFIG_PATH`, then run the config-generation cell. It writes a
portable project-relative config whose `cache_dir` points to the requested dataset.
Set `RUN_DOWNLOAD = True` only after `ENV_FILE` contains valid CDSE credentials.
The acquisition saves weather and satellite inputs, quality flags, catalog metadata,
raw responses and checksums; it refuses to overwrite an existing destination.

After the download, inspect the manifest and run:

```powershell
.\.venv-win\Scripts\python.exe -m root_zone_water.runner --config config_my_dataset.json --output outputs/my-dataset
```

To use that dataset in the main notebook, copy the generated dates, filter and
`cache_dir` into `config.json`, then run `execute_notebook.py`. The generated config
does not contain credentials.
