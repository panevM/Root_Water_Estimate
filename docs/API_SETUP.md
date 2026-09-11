# Offline setup and API acquisition

Cached mode requires Python dependencies only. It does not read environment files or call APIs.
Install `requirements-tested.txt` for the versions actually used (or `requirements.txt` for
compatible ranges). The checked-in weather responses are genuine, sanitized API responses.

## Copernicus Data Space Ecosystem (CDSE)

1. Register at [Copernicus Data Space](https://dataspace.copernicus.eu/).
2. Follow the official [Sentinel Hub authentication guide](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html).
   Open the [Sentinel Hub dashboard](https://shapps.dataspace.copernicus.eu/dashboard/), User Settings,
   OAuth clients, Create. Create a CDSE OAuth client and securely retain its secret.
3. Copy `.env.example` to `.env` locally and set **SENTINEL_CLIENT_ID** and
   **SENTINEL_CLIENT_SECRET**. These exact names are used by the local client. The file
   is ignored. Do not paste secrets into notebooks or commit them.
4. The client-credentials token endpoint is
   `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`.
   `SentinelClient` reuses tokens until shortly before expiry. Headers/token responses are never cached.

The client retrieves Sentinel-2 products through a data service; it does not communicate with a
satellite. See the [CDSE Statistical API](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Statistical.html),
[Statistical API details](https://docs.sentinel-hub.com/api/latest/api/statistical/), and
[Catalog API](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html).

## Small connection and contract check

From `robotics2/`, use the environment's Python:

```powershell
python connection_check.py --env-file .env --date 2025-06-10 --output data/refresh-check
```

This retrieves a single complete UTC day, with catalog reconciliation and metric sample-count
checks through the same adapter. An empty/cloudy day is not an authentication success with usable
observations: inspect the manifest and choose another catalog acquisition day if needed.
The checked-in dataset demonstrates the offline processing contract. A personal capture is still
needed to validate the current CDSE account, quota and live response for a user's chosen interval.
Mocked request tests do not establish live API compatibility.

## Capture the complete interval

For an interactive workflow, open `Create_Custom_Dataset.ipynb`. Edit the five inputs
in its configuration cell, run it to create `config_my_dataset.json`, inspect the
values, then set `RUN_DOWNLOAD = True` to call the existing acquisition client. The
notebook writes the dataset path into `cache_dir` before downloading, so the config
and dataset remain paired.

```powershell
python -m root_zone_water.acquisition --env-file .env --output data/refresh-wheat-2025
```

Then inspect `data/refresh-wheat-2025/manifest.json`. It must report complete weather and usable
NDMI days. Inspect SCL vegetation context and field suitability. Review counts, raw responses,
checksums and calendar assumptions before changing `cache_dir` in `config.json` to this directory.
Run `python execute_notebook.py` again. The acquisition command refuses an existing destination;
refresh never overwrites the published dataset. API mode is also available through
`load_data(config, mode='api', destination='data/refresh-new')` with the same raw processors.
No credential loading occurs implicitly; use `load_dotenv` explicitly for that Python API if needed.

The generated config can be used directly without editing the main config:

```powershell
python -m root_zone_water.runner --config config_my_dataset.json --output outputs/my-dataset
```

## Open-Meteo

The [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) public
noncommercial endpoint requires no key; check the provider's current terms and limits before
commercial or large-volume use. Attribution: weather data from Open-Meteo, ERA5 reanalysis from
ECMWF/Copernicus. See [Open-Meteo terms](https://open-meteo.com/en/terms).

We explicitly request `models=era5`, UTC, and daily precipitation, rain, snowfall, ET0 and minimum
temperature. Returned grid coordinates/elevation can differ from the polygon centre. ERA5 is about
0.25 degrees; these are reanalysis grid inputs, not an on-field rain gauge. ET0 is supplied directly
by Open-Meteo using its FAO reference-evapotranspiration implementation; do not calculate ET0 again.
ERA5 historical data have publication lag; this example intentionally uses 2025, not recent days.
For another interval, check actual availability and do not substitute forecast or fixed weather silently.

The acquisition requests explicit ERA5 weather fields and validates completeness before writing the
dataset. If a provider response returns null required fields, acquisition rejects that response rather
than silently substituting values.

## Troubleshooting

- Missing credentials: provide an environment file with the two exact names via `--env-file`.
- 401: invalid/expired client credentials; verify the CDSE client, then retry a new capture.
- 403: check account permission and available quota. 429: rate limit; wait before another capture.
- 400/schema error: compare saved sanitized requests with the current official API reference.
- Empty catalog/statistics: inspect interval and polygon; P1D bins do not imply daily imagery.
- Low coverage: masked cloud/shadow/snow or invalid band data; do not treat as zero index values.
- Multiple acquisition times: this implementation conservatively rejects that day's correction;
  extend a documented compositing policy before accepting it.
- Missing weather: stop. Never replace it with zero precipitation or fixed ET0.
- Snowfall or freezing: the single liquid compartment rejects the interval. Choose an appropriate
  snow-free active-growth interval; do not silently count snow as liquid water.
- Network/timeout: three bounded attempts, 15 s connect/90 s response timeout; only transient
  connection, 429 and 5xx failures retry. Error output omits credentials and server bodies.

The remaining user-specific action is to provide a valid CDSE OAuth client, choose an interval and
dataset directory in `Create_Custom_Dataset.ipynb`, capture the data, inspect its real scene/sample
metadata, and then run the estimator with the generated config.
