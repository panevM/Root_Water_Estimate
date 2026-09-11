# Dataset provenance and dictionary

## Captured data and gaps

`data/refresh-wheat-2025-90d/` contains genuine Open-Meteo ERA5 weather for 90 inclusive dates,
2025-05-01 to 2025-07-29, and Sentinel-2 daily records for the same study interval. The saved
satellite table has 45 records, of which 26 pass the NDMI quality policy and 19 remain rejected
with reasons. An empty or rejected satellite row is never replaced with a generated observation.
No simulator, database output or fake ground truth is used. The notebook's water states are
speculative model/filter outputs, not field observations.

`data/teaching-2025/` retains the initial ERA5-Land response with missing required inputs.
`data/attempt-sandbox/` records the network-restricted failed attempt. Neither is used in analysis.
`Create_Custom_Dataset.ipynb` provides the supported workflow for creating another self-contained
dataset and writing a matching config. The exact credential and acquisition details are in
`API_SETUP.md`.

## Area and crop

Requested WGS84 polygon: `(21.43,41.99), (21.44,41.99), (21.44,42.00), (21.43,42.00), (21.43,41.99)`.
Weather requested at the arithmetic vertex centre, 41.995 N, 21.435 E. This is the original configured
rectangle, not a surveyed wheat field. The approximately 0.92 km² boundary near Skopje may mix land
uses. No imagery has been acquired to verify suitability. The optional SCL vegetation share in a
future capture is context, not crop identification: low vegetation should preclude representative
crop interpretation; even high vegetation does not verify wheat. No seeded farm hectares are used.

Assumed crop: spring wheat, planting 2025-04-01. Initial/development/mid/late durations 20/30/60/40
days use the FAO56 small-grain Mediterranean April example as a teaching transfer, not local
phenological evidence. Study dates cover development, mid and late stages. Root depth is fixed at
1.4 m even early in the window; this intentionally omits growth in effective rooting depth.
Soil silt FC/WP=.32/.15 are reference selections; saturation=.46 and drainage=.10/day are explicit
teaching assumptions. Irrigation is assumed zero, and no recommendation enters the runner.

## Time, weather and physical accounting

Daily intervals are `[00:00 UTC, next 00:00 UTC)`. Weather is retrospectively aggregated for the
whole day. A catalog-timed image is assigned to that day's closing boundary, after prediction;
this is a timing approximation, not strict forecasting. NDVI from that image first becomes eligible
as Kc forcing on the next interval. Hold the last accepted NDVI for at most 15 days measured from
actual acquisition time; otherwise use calendar Kc. Missing NDMI always means prediction only.

Precipitation includes rain/showers/snow water equivalent. We retain precipitation, rain and snow
separately, reject any snowfall or freezing minimum temperature, and then use precipitation as
liquid input. There is no snowpack, antecedent snowmelt or frozen-soil model. No observed snowfall
within the interval is not proof of no antecedent snow; that remains an active-season assumption.
Effectiveness defaults to 1; other choices record ineffective precipitation as an explicit loss.
Infiltration is assumed immediate up to capacity, with named overflow. Interception, capillary rise,
infiltration-rate limits and lateral redistribution are omitted. ET cannot extract below Wwp.
Drainage is nonnegative after ET. Physical balance is evaluated at each interval's starting mean.
The UKF expectation can differ from that physical mean trajectory; its distribution adjustment is
logged separately, as is the assimilation adjustment. Neither is rainfall or irrigation.

## Files and metadata

| File | Contents |
|---|---|
| manifest.json | Processing version; retrieval time; completeness/errors; exact config; URLs; returned weather metadata; SHA-256 checksums |
| geometry.geojson | Original lon/lat closed polygon |
| config.snapshot.json | Acquisition-time crop/calendar/soil/noise/geometry assumptions |
| raw/weather.request.json | Exact public query: centre, dates, daily fields, UTC, explicit model |
| raw/weather.json | Entire genuine weather response: returned coordinates, elevation, timezone, units, arrays |
| weather.csv | Compact table rebuilt from raw response in cached mode |
| raw/catalog.json (future capture) | Paginated product IDs, actual scene times, product metadata; OAuth excluded |
| raw/{ndvi,ndmi,vegetation}.request.json (future) | Polygon projected to EPSG:32634; 20 m grid; nearest resampling; evalscript; complete intervals |
| raw/{ndvi,ndmi,vegetation}.json (future) | Full statistics sufficient to rebuild compact satellite table |
| satellite.csv (future) | Sparse interval stats, actual times/IDs, validity/rejection policy |

Cache loader verifies all manifest hashes, rebuilds tables using the same raw processors as API
mode and rejects acquisition-setting mismatches. Noise/soil/proxy settings may vary for explicit
sensitivity runs; thresholds and uncertainty scales recompute. Refresh writes a new directory.
The manifest fingerprints every file in the self-contained dataset.

Weather fields: `date` UTC interval label; `precipitation_mm`, `rain_mm`, `et0_mm` daily integrated
depths; `snowfall_cm` snow depth amount; `temperature_min_c` daily minimum. No spatial field precision
is implied by their reported decimal places. Open-Meteo's FAO reference ET0 approximates demand
over well-watered reference grass, using meteorological radiation/temperature/humidity/wind; Kc
transfers potential demand to the assumed crop. ET0 is not actual wheat evapotranspiration.

## Sentinel processing contract

NDVI=(B08-B04)/(B08+B04+1e-6); NDMI=(B8A-B11)/(B8A+B11+1e-6). Primary aggregation is the
arithmetic mean of valid **per-pixel indices**. See the [NDMI band reference](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndmi/).
Indices are dimensionless. The two bands of each ratio must be finite/nonnegative and have a sum
above 1e-6. No-data and SCL 3/8/9/10/11 (shadow/cloud/cirrus/snow) remain excluded, with SCL 0/1
(no data/saturated-defective) additionally rejected. Accepted 2/4/5/6/7 include dark/topographic
shadow, vegetation, bare/nonvegetated, water and unclassified surfaces. This masking is not a wheat
or agricultural land mask; leastCC does not mean cloud-free area pixels.

Separate requests preserve independent NDVI/NDMI validity. `sampleCount-noDataCount` gives valid
samples; minimum 100 samples and fraction .50 are required independently. `sampleCount` can include
outside-polygon bounding-grid cells, so this fraction is deliberately conservative. Projected geometry
is sent directly, never replaced by a bbox. Expected ~2300 area pixels and surrounding-grid count
are recorded and checked approximately; boundary rasterization can differ. Nearest-neighbour
resampling explicitly puts B08/B04 and B8A/B11/SCL on a common 20 m grid. Samples are spatially
correlated; resampling does not create independent evidence. Live returned sample counts remain
unverified until credentials permit a real response.

Catalog IDs are deduplicated; more than one distinct acquisition time in a day is conservatively
rejected, even if statistics exist. A single-time set of overlapping products is one leastCC mosaic,
assimilated once. IDs list contributing candidates, not proven per-pixel source attribution. Preserve
all catalog timestamps and distinguish them from interval dates. If provider metadata cannot
reconcile a day, no correction or NDVI input is accepted. P1D is a bin size, not availability.

Satellite fields: `date`; `product_ids`/`observation_id` semicolon-separated catalog candidate IDs;
`acquisition_times_utc`, `acquisition_time_utc` (only a unique time); `catalog_reconciled`;
`ndvi`, `ndmi`, `vegetation` (SCL4 share among accepted pixels); each has `*_spatial_std`,
`*_sample_count`, `*_valid_count`, `*_valid_fraction`, `*_accepted`; `rejection_reason`.
Rejected and empty intervals are preserved with reasons, never forward-filled for correction.

## Uncertainty and output dictionary

W/P/R_t/Q_t units are mm/mm²/mm²/mm². Configuration exposes standard deviations; squared values
form scalar covariances (equivalent to 1x1 matrices). In course notation R_t is process covariance
and Q_t is measurement covariance. Initial midpoint W=329 mm; standard deviation
TAW/sqrt(12)=68.705 mm approximates a uniform prior over [Wwp,Wfc]. Process discrepancy standard
deviation .02*TAW=4.76 mm/day; proxy model discrepancy .01*TAW=2.38 mm for the configured
silt/wheat teaching parameters. These are broad, editable teaching assumptions, not fitted
uncertainty. The NDMI standard-deviation floor .05 is dimensionless.
If enabled, larger spatial pixel dispersion replaces that floor as a conservative **heuristic**,
without division by sqrt(pixel count). Missing spatial statistics retain the floor and model discrepancy.

Interior propagation uses sigma_index_water=TAW/(mwet-mdry)*sigma_NDMI. Q_t adds its square to
sigma_proxy_model². Endpoint clipping doubles Q_t instead of pretending uncertainty disappears.
Clipping censors the index; the Gaussian likelihood is approximate there. Systematic endpoint or
canopy bias is not cured by wide bands. NDVI forcing and NDMI correction from the same image may
share errors; the filter neglects that cross-correlation, another conditional assumption.

R_t adds daily process discrepancy and optional squared finite-difference sensitivities to precipitation
and ET0 times their variances (1 mm and .5 mm standard deviations). Assumed independence and local
linear propagation omit correlated weather/model errors. The discrepancy allowance is defined as
remaining dynamics error; weather uncertainty is accounted separately to avoid intentional double counting.

Daily logs include UTC interval boundaries; forcing/Kc/stage/source/NDVI age; start storage and P;
physical prediction and named water fluxes; prior/posterior means and covariances; R_t and sensitivities;
predicted proxy h(Wprior)=Wprior; actual heuristic proxy/Q_t/clipped flag; pre-correction innovation,
S=Pprior+Q_t, gain; update/rejection flags and reason; observation age (boundary minus actual acquisition);
assimilation and UKF distribution adjustments; Jacobian/sigma-point clipping and bound safeguards.
Missing diagnostics are NaN, never automatic zero residuals. Prior MAE/RMSE mean disagreement with
the heuristic proxy in mm, never root-zone accuracy. No posterior-proxy performance score is reported.
