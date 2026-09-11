"""Explicit API-only acquisition; adapted from active services clients, no eager IO."""
import argparse
import hashlib
import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import requests
from pyproj import Transformer, CRS
from .config import ROOT, load_config, parameters

VERSION = 'root-zone-water-1'
TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
STATS_URL = 'https://sh.dataspace.copernicus.eu/api/v1/statistics'
CATALOG_URL = 'https://sh.dataspace.copernicus.eu/catalog/v1/search'
WEATHER_URL = 'https://archive-api.open-meteo.com/v1/archive'

class AcquisitionError(RuntimeError): pass

def request_json(method, url, **kwargs):
    for attempt in range(3):
        try:
            response = requests.request(method, url, timeout=(15, 90), **kwargs)
        except requests.RequestException:
            if attempt == 2: raise AcquisitionError('Network/timeout failure at ' + url) from None
            time.sleep(2**attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(2**attempt)
            continue
        if response.status_code != 200:
            # Never log request headers, credentials, token response or server bodies.
            label = {401:'authentication',403:'permission/quota',429:'quota/rate limit'}.get(response.status_code,'request/schema')
            raise AcquisitionError(f'{label} error HTTP {response.status_code} at {url}')
        try: return response.json()
        except ValueError: raise AcquisitionError('Invalid JSON response at ' + url) from None

class SentinelClient:
    def __init__(self): self.token, self.expires = None, 0

    def headers(self):
        if time.monotonic() >= self.expires:
            cid, secret = os.getenv('SENTINEL_CLIENT_ID'), os.getenv('SENTINEL_CLIENT_SECRET')
            if not cid or not secret:
                raise AcquisitionError('Missing SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET; see docs/API_SETUP.md')
            raw = request_json('POST', TOKEN_URL, data=dict(grant_type='client_credentials',client_id=cid,client_secret=secret))
            if not raw.get('access_token'): raise AcquisitionError('Token response lacks access_token')
            self.token = raw['access_token']
            self.expires = time.monotonic() + max(1, float(raw.get('expires_in',300))-30)
        return {'Authorization': 'Bearer '+self.token}

    def catalog(self, c):
        start,end = interval(c)
        query = dict(collections=['sentinel-2-l2a'], intersects=c['geometry'],
                     datetime=start+'/'+end, limit=100)
        pages, features = [], []
        for _ in range(100):
            page = request_json('POST',CATALOG_URL,headers=self.headers(),json=query)
            pages.append(page)
            features.extend(page.get('features',[]))
            nxt = page.get('context',{}).get('next')
            if nxt is None: break
            query['next'] = nxt
        else: raise AcquisitionError('Catalog exceeded bounded pagination')
        ids = [f['id'] for f in features]
        if len(ids) != len(set(ids)): raise AcquisitionError('Duplicate catalog products')
        return pages, features

def interval(c):
    return c['start_date']+'T00:00:00Z', (date.fromisoformat(c['end_date'])+timedelta(days=1)).isoformat()+'T00:00:00Z'

def projected_geometry(c):
    crs = CRS.from_epsg(c['projected_epsg'])
    if not crs.is_projected or any(axis.unit_name != 'metre' for axis in crs.axis_info):
        raise ValueError('Statistical resolution requires a projected CRS in metres')
    geometry = c['geometry']
    if geometry.get('type') != 'Polygon' or not geometry.get('coordinates'):
        raise ValueError('Expected GeoJSON Polygon')
    for ring in geometry['coordinates']:
        if len(ring) < 4 or ring[0] != ring[-1]: raise ValueError('Polygon rings must be closed')
        if any(len(point)!=2 or not all(math.isfinite(x) for x in point) or
               not (-180<=point[0]<=180 and -90<=point[1]<=90) for point in ring):
            raise ValueError('Invalid lon/lat polygon coordinates')
    transform = Transformer.from_crs(4326,c['projected_epsg'],always_xy=True)
    return {'type':'Polygon','coordinates':[[list(transform.transform(*point)) for point in ring] for ring in c['geometry']['coordinates']]}

def grid_metadata(c):
    rings = projected_geometry(c)['coordinates']
    ring = rings[0]
    width = max(x for x,y in ring)-min(x for x,y in ring)
    height = max(y for x,y in ring)-min(y for x,y in ring)
    areas = [abs(sum(r[i][0]*r[i+1][1]-r[i+1][0]*r[i][1] for i in range(len(r)-1)))/2 for r in rings]
    area = areas[0]-sum(areas[1:])
    return dict(projected_epsg=c['projected_epsg'],resolution_m=c['resolution_m'],
                polygon_area_m2=area,expected_polygon_pixels_approx=area/c['resolution_m']**2,
                bbox_grid_pixels_approx=math.ceil(width/c['resolution_m'])*math.ceil(height/c['resolution_m']),
                sample_count_note='sampleCount includes noDataCount and bounding-grid pixels outside polygon; fractions use that conservative denominator')

def evalscript(index):
    # Separate calls let valid NDVI survive invalid NDMI and vice versa.
    if index not in ('ndvi','ndmi','vegetation'): raise ValueError(index)
    a,b = ('B08','B04') if index == 'ndvi' else ('B8A','B11')
    expression = '(s.SCL===4 ? 1 : 0)' if index == 'vegetation' else f'(s.{a}-s.{b})/(s.{a}+s.{b}+1e-6)'
    guard = 'true' if index == 'vegetation' else f'Number.isFinite(s.{a}) && Number.isFinite(s.{b}) && s.{a}>=0 && s.{b}>=0 && Math.abs(s.{a}+s.{b})>1e-6'
    return '''//VERSION=3
function setup() {return {input:[{bands:["B04","B08","B8A","B11","SCL","dataMask"]}],
output:[{id:"value",bands:1,sampleType:"FLOAT32"},{id:"dataMask",bands:1}]};}
function evaluatePixel(s) {
 let valid = s.dataMask===1 && ![0,1,3,8,9,10,11].includes(s.SCL) && GUARD;
 return {value:[valid ? EXPRESSION : 0],dataMask:[valid ? 1 : 0]};
}'''.replace('GUARD',guard).replace('EXPRESSION',expression)

def statistics_request(c, index):
    start,end = interval(c)
    return {'input':{'bounds':{'geometry':projected_geometry(c),'properties':{'crs':f"http://www.opengis.net/def/crs/EPSG/0/{c['projected_epsg']}"}},
        'data':[{'type':'sentinel-2-l2a','dataFilter':{'mosaickingOrder':'leastCC'},
                 'processing':{'upsampling':'NEAREST','downsampling':'NEAREST'}}]},
        'aggregation':{'timeRange':{'from':start,'to':end},'aggregationInterval':{'of':'P1D'},
                       'resx':c['resolution_m'],'resy':c['resolution_m'],'evalscript':evalscript(index)},
        'calculations':{'value':{}}}

def weather_request(c):
    ring = c['geometry']['coordinates'][0][:-1]
    return dict(latitude=sum(p[1] for p in ring)/len(ring),longitude=sum(p[0] for p in ring)/len(ring),
                start_date=c['start_date'],end_date=c['end_date'],timezone='UTC',models=c['weather_model'],
                daily='precipitation_sum,rain_sum,snowfall_sum,et0_fao_evapotranspiration,temperature_2m_min')

def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def acquire(c, destination, weather_only=False):
    """New directory only: never silently replace a published teaching cache."""
    from .data import weather_table, satellite_table
    destination = Path(destination)
    if destination.exists(): raise AcquisitionError('Destination exists; choose a new cache directory')
    destination.mkdir(parents=True)
    manifest = dict(processing_version=VERSION,retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                    status='incomplete',config=c,weather_source=WEATHER_URL,satellite_source=STATS_URL,
                    catalog_source=CATALOG_URL,errors=[])
    manifest['grid'] = grid_metadata(c)
    manifest['reference_selections'] = parameters(c)[1:]
    dump(destination/'geometry.geojson', c['geometry'])
    dump(destination/'config.snapshot.json',c)
    try:
        query = weather_request(c)
        raw = request_json('GET',WEATHER_URL,params=query)
        dump(destination/'raw/weather.json',raw)
        dump(destination/'raw/weather.request.json',query)
        weather = weather_table(raw,c)
        weather.to_csv(destination/'weather.csv',index=False)
        manifest['weather_days'] = len(weather)
        manifest['weather_returned_metadata'] = {k:v for k,v in raw.items() if k not in ('daily','generationtime_ms')}
    except (AcquisitionError,ValueError) as e:
        manifest['errors'].append(str(e))
    if not weather_only:
        try:
            client = SentinelClient()
            pages,features = client.catalog(c)
            dump(destination/'raw/catalog.json',pages)
            if not features: raise AcquisitionError('Catalog returned no scenes')
            raw_stats = {}
            for name in ('ndvi','ndmi','vegetation'):
                body = statistics_request(c,name)
                dump(destination/f'raw/{name}.request.json',body)
                raw_stats[name] = request_json('POST',STATS_URL,headers=client.headers(),json=body)
                dump(destination/f'raw/{name}.json',raw_stats[name])
            satellite = satellite_table(raw_stats,features,c)
            satellite.to_csv(destination/'satellite.csv',index=False)
            manifest['satellite_rows'] = len(satellite)
            manifest['usable_ndmi_days'] = int(satellite['ndmi_accepted'].sum())
            if not manifest['usable_ndmi_days']: raise AcquisitionError('No usable NDMI after coverage/catalog policy')
        except (AcquisitionError,ValueError) as e:
            manifest['errors'].append(str(e))
    else: manifest['errors'].append('Satellite acquisition explicitly skipped (--weather-only)')
    if not manifest['errors']: manifest['status']='complete'
    manifest['files_sha256'] = {str(f.relative_to(destination)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(destination.rglob('*')) if f.is_file()}
    dump(destination/'manifest.json',manifest)
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--weather-only',action='store_true')
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=False)
    result = acquire(load_config(args.config),args.output,args.weather_only)
    print(json.dumps({k:result.get(k) for k in ('status','weather_days','usable_ndmi_days','errors')},indent=2))
    if result['status'] != 'complete': raise SystemExit(2)

if __name__ == '__main__': main()
