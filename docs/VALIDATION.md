# Executed validation — 2026-09-10

- Standard CPython environment; exact installed versions in `requirements-tested.txt`.
- `python -m pytest -q`: **25 passed**. Focus: physical conservation, thresholds/root scaling,
  shared proxy/censoring, analytical water-balance Jacobian, scalar affine EKF/UKF agreement, course covariance correction, bounded
  sigma points, daily uniqueness, independent NDVI/NDMI validity, causal NDVI age/fallback,
  missing/frozen weather, UTC intervals, projected grid/sample counts and token reuse.
- `python -m root_zone_water.runner`: all three filter runs completed the configured interval. Zero irrigation on every
  day. Maximum absolute physical budget residual: 0.0 mm. Real-data corrections and proxy pairs are
  reported in the generated diagnostics.
- `python execute_notebook.py`: fresh-kernel Run All completed with outbound Python socket
  connections blocked. All five code cells executed, zero error outputs, five saved figures.
  Figures inspected; area tick spacing and crop-stage labels corrected and notebook re-executed.
- Cached inputs: 92 daily weather records; total precipitation 70.70 mm, reference ET0 501.73 mm;
  minimum daily minimum temperature 6.3 °C; snowfall 0 cm. Requested centre 41.995 N / 21.435 E;
  returned location/grid metadata remain in the raw response and manifest.
- Projected area approximately 919,615 m², approximately 2,299 polygon pixels at 20 m;
  bounding grid approximately 2,352 pixels. These are calculated expectations, **not validated
  returned Sentinel sample counts**.
- No production application files edited by this implementation. Reused source fingerprints in
  the saved dataset manifest; the numerical package has no external application or cache imports.

**Incomplete requirement:** no genuine satellite scene statistics were captured. Credential lookup
found neither Sentinel variable in the environment nor a configured local `.env`. Acquisition
recorded the explicit missing-credential error. The Statistical API and catalog contracts are
implemented and tested with small unit fixtures, but live authenticated compatibility, pixel counts,
field suitability, real innovations and observation-based sensitivities remain unverified.

The available cached Run All is a weather-driven conditional estimator demonstration, not a
completed genuine satellite-assimilation study. Follow `API_SETUP.md` to capture and select a new
complete cache. No synthetic observations or purported ground truth were generated.
