"""Same raw-response processors for API capture and offline replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT

def dates(c): return pd.date_range(c['start_date'],c['end_date'],freq='D').strftime('%Y-%m-%d').tolist()

def weather_table(raw,c):
    mapping = {'precipitation_sum':'precipitation_mm','rain_sum':'rain_mm','snowfall_sum':'snowfall_cm',
               'et0_fao_evapotranspiration':'et0_mm','temperature_2m_min':'temperature_min_c'}
    if raw.get('utc_offset_seconds') != 0: raise ValueError('Weather must use UTC')
    units = raw.get('daily_units',{})
    for key,unit in dict(precipitation_sum='mm',rain_sum='mm',snowfall_sum='cm',
                         et0_fao_evapotranspiration='mm',temperature_2m_min='°C').items():
        if units.get(key) != unit: raise ValueError('Unexpected weather units for '+key)
    daily = raw.get('daily',{})
    try: result = pd.DataFrame({'date':daily['time'],**{v:daily[k] for k,v in mapping.items()}})
    except (KeyError,ValueError): raise ValueError('Missing/inconsistent daily weather arrays') from None
    return validate_weather(result,c)

def validate_weather(result,c):
    result = result.sort_values('date').reset_index(drop=True)
    if result['date'].duplicated().any() or result['date'].tolist() != dates(c):
        raise ValueError('Weather dates must be unique and cover the entire study interval')
    cols = ['precipitation_mm','rain_mm','snowfall_cm','et0_mm','temperature_min_c']
    if not np.isfinite(result[cols].to_numpy(float)).all(): raise ValueError('Missing/nonfinite required weather')
    if (result[cols[:-1]] < 0).any().any(): raise ValueError('Negative weather amount')
    # Refuse unsupported snow/freeze rather than inventing a snowpack or snowmelt flux.
    if (result['snowfall_cm'] > 0).any() or (result['temperature_min_c'] <= 0).any():
        raise ValueError('Snowfall/freezing in interval: choose a snow-free active-growth interval; snowpack is unsupported')
    return result

def _stats(raw):
    result = {}
    for item in raw.get('data',[]):
        if 'error' in item: raise ValueError('Statistical interval contains an API error')
        day = item['interval']['from'][:10]
        if day in result: raise ValueError('Duplicate Statistical API day')
        bands = item.get('outputs',{}).get('value',{}).get('bands',{})
        stats = next(iter(bands.values()),{}).get('stats',{})
        result[day] = (item['interval'],stats)
    return result

def _finite_float(value):
    """Normalize Statistical API numeric values, including JSON "NaN" strings."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None

def satellite_table(raws,features,c):
    from .acquisition import grid_metadata
    grid = grid_metadata(c)
    if len({f['id'] for f in features}) != len(features): raise ValueError('Duplicate catalog products')
    by_day = {}
    for f in features:
        ts = f['properties']['datetime']
        by_day.setdefault(ts[:10],[]).append((f['id'],ts))
    parsed = {name:_stats(raw) for name,raw in raws.items()}
    records = []
    for day in sorted(set().union(*(set(v) for v in parsed.values()))):
        if day not in dates(c): raise ValueError('Satellite date outside requested interval')
        candidates = sorted(by_day.get(day,[]))
        times = sorted(set(ts for _,ts in candidates))
        # Reject multi-time daily mosaics so no guessed acquisition time or repeated corrections.
        reconciled = len(times)==1
        row = dict(date=day,product_ids=';'.join(pid for pid,_ in candidates),
                   acquisition_times_utc=';'.join(times),acquisition_time_utc=times[0] if reconciled else '',
                   catalog_reconciled=reconciled,observation_id=';'.join(pid for pid,_ in candidates))
        for name in ('ndvi','ndmi','vegetation'):
            inter,s = parsed[name].get(day,({},{}))
            if inter:
                expected_end = (pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
                if pd.Timestamp(inter['from']) != pd.Timestamp(day+'T00:00:00Z') or pd.Timestamp(inter['to']) != pd.Timestamp(expected_end):
                    raise ValueError('Unexpected incomplete satellite interval')
            total,missing = s.get('sampleCount',0),s.get('noDataCount',0)
            total,missing = int(total or 0),int(missing or 0)
            if not (0 <= missing <= total): raise ValueError('Invalid sample counts')
            if total and not (.5*grid['expected_polygon_pixels_approx'] <= total <= 1.25*grid['bbox_grid_pixels_approx']):
                raise ValueError('Returned sampleCount inconsistent with requested metric grid')
            valid = total-missing
            fraction = valid/total if total else 0
            mean,std = _finite_float(s.get('mean')), _finite_float(s.get('stDev'))
            finite = mean is not None and -1 <= mean <= 1
            accepted = finite and reconciled and fraction >= c['min_valid_fraction'] and valid >= c['min_valid_pixels']
            row.update({name:mean if finite else np.nan,name+'_spatial_std':std,
                        name+'_sample_count':total,name+'_valid_count':valid,name+'_valid_fraction':fraction,
                        name+'_accepted':bool(accepted)})
        row['rejection_reason'] = ('unreconciled_or_multiple_acquisitions' if not reconciled else
                                   'insufficient_valid_ndmi' if not row['ndmi_accepted'] else '')
        records.append(row)
    if not records: raise ValueError('Statistical API returned no intervals')
    return pd.DataFrame(records)

def load_data(c, mode=None, destination=None):
    mode = mode or c['mode']
    if mode == 'api':
        if destination is None: raise ValueError('API mode requires a new explicit destination')
        from .acquisition import acquire
        acquire(c,destination)
        path = Path(destination)
    elif mode == 'cached': path = ROOT / c['cache_dir']
    else: raise ValueError('mode must be cached or api')
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != digest: raise ValueError('Cache checksum mismatch: '+name)
    for key in ('geometry','start_date','end_date','timezone','weather_model','resolution_m','projected_epsg'):
        if c[key] != manifest['config'][key]: raise ValueError('Cache/config mismatch: '+key)
    weather = weather_table(json.loads((path/'raw/weather.json').read_text(encoding='utf-8')),c)
    if (path/'raw/ndmi.json').exists():
        raws = {k:json.loads((path/f'raw/{k}.json').read_text(encoding='utf-8')) for k in ('ndvi','ndmi','vegetation')}
        features = [f for page in json.loads((path/'raw/catalog.json').read_text(encoding='utf-8')) for f in page['features']]
        satellite = satellite_table(raws,features,c)
    else:
        # Empty structure describes missing acquisition; it is never a generated observation dataset.
        satellite = pd.DataFrame(columns=['date','ndvi','ndmi','ndmi_spatial_std','ndvi_accepted','ndmi_accepted',
                                          'acquisition_time_utc','observation_id','rejection_reason'])
    return weather,satellite,manifest
