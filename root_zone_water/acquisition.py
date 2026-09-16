"""Експлицитно преземање од Open-Meteo и Copernicus во нов локален кеш.

acquire е влезот за CLI, connection_check и data.load_data во режим api.
Ги зачувува барањата, суровите одговори, обработените табели и manifest;
data.py ги врши истите проверки и при подоцнежно читање без мрежа.
Самото увезување на модулот не прави мрежни барања или датотеки.
"""
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

class AcquisitionError(RuntimeError):
    """Грешка при мрежно преземање, автентикација или проверка на преземањето."""
    pass

def request_json(method, url, **kwargs):
    """Испрати HTTP method до url и врати декодиран JSON од одговор со статус 200.

    acquire и SentinelClient предаваат headers, json, params или data преку
    kwargs. Има најмногу три обиди, чекање 1/2 секунди меѓу повторувањата
    и временски ограничувања 15 s за поврзување, 90 s за читање.
    Повторува мрежни грешки и HTTP 429/500/502/503/504; друг статус или
    невалиден JSON крева AcquisitionError без откривање тајни или тело.
    """
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
            # Не се испишуваат заглавија, лозинки, токени или тела на серверски одговори.
            label = {401:'authentication',403:'permission/quota',429:'quota/rate limit'}.get(response.status_code,'request/schema')
            raise AcquisitionError(f'{label} error HTTP {response.status_code} at {url}')
        try: return response.json()
        except ValueError: raise AcquisitionError('Invalid JSON response at ' + url) from None

class SentinelClient:
    """Клиент за каталог и повторна употреба на OAuth токен во едно acquire."""
    def __init__(self):
        """Постави празен токен и истечен рок; првото headers ќе ја побара мрежата."""
        self.token, self.expires = None, 0

    def headers(self):
        """Врати Authorization заглавие, обновувајќи го зачуваниот токен по потреба.

        Го користат catalog и статистичките барања во acquire. Ги чита
        SENTINEL_CLIENT_ID и SENTINEL_CLIENT_SECRET од околината и може да направи POST за
        токен. Ги менува self.token/self.expires; рокот е во монотони секунди
        со резерва од 30 s (најмалку 1 s). Отсутни тајни или токен креваат
        AcquisitionError. Не го запишува токенот на диск.
        """
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
        """Врати (сурови страници, обединети features) за геометријата и периодот во c.

        acquire го повикува за Sentinel-2 L2A: POST барања со најмногу
        100 страници по 100 продукти и headers со обновлив токен.
        Празен каталог враќа празна листа; дупликати на id или продолжување
        по лимитот креваат AcquisitionError. Нема запишување датотеки.
        """
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
    """Врати (почетна полноќ, полноќ по end_date) како ISO UTC стрингови.

    catalog/statistics_request го користат c со YYYY-MM-DD датуми;
    крајот ја опфаќа целата последна зададена дата за дневната агрегација.
    """
    return c['start_date']+'T00:00:00Z', (date.fromisoformat(c['end_date'])+timedelta(days=1)).isoformat()+'T00:00:00Z'

def projected_geometry(c):
    """Врати GeoJSON Polygon од c['geometry'] проектиран во c['projected_epsg'].

    statistics_request/grid_metadata бараат координати во метри, наместо
    степени. Влезните затворени прстени се (должина, ширина) во EPSG:4326,
    со опсези [-180,180]/[-90,90]. Невалиден полигон или CRS што не е
    проектиран во метри крева ValueError. Не ја менува c и нема мрежен повик.
    """
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
    """Пресметај приближна површина и број пиксели за барањето и проверките.

    c ги дава полигонот, projected_epsg и resolution_m. Враќа речник со
    polygon_area_m2 (надворешна површина минус дупки),
    expected_polygon_pixels_approx (површина/резолуција²),
    bbox_grid_pixels_approx (опишан правоаголник) и метаподатоци за мрежата.
    satellite_table ги користи како груба проверка на sampleCount;
    ова не е растеризација и не брои точно валидни пиксели.
    """
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
    """Состави JavaScript за statistics_request; index е ndvi, ndmi или vegetation.

    NDVI користи B08/B04, NDMI B8A/B11 со (a−b)/(a+b+1e-6).
    Се бараат конечни ненегативни рефлектанси и |a+b|>1e-6.
    dataMask мора да е 1; SCL класите [0,1,3,8,9,10,11] се исклучуваат.
    vegetation враќа 1 за SCL=4 и 0 за другите дозволени класи, за удел
    на вегетација. Враќа код, не го извршува; друго име крева ValueError.
    Директивата //VERSION=3 е дел од API договорот и мора да се зачува.
    """
    # Одделните барања овозможуваат независно прифаќање на NDVI и NDMI.
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
    """Врати тело на Statistical API барање за acquire, без испраќање.

    c задава проектиран полигон, период и resolution_m (проверени 20 m),
    а index го избира evalscript. Сите канали користат иста метричка мрежа,
    најблизок сосед и дневни UTC интервали P1D; mosaickingOrder е leastCC.
    Усогласувањето со вистинските времиња на снимките следи во satellite_table.
    """
    start,end = interval(c)
    return {'input':{'bounds':{'geometry':projected_geometry(c),'properties':{'crs':f"http://www.opengis.net/def/crs/EPSG/0/{c['projected_epsg']}"}},
        'data':[{'type':'sentinel-2-l2a','dataFilter':{'mosaickingOrder':'leastCC'},
                 'processing':{'upsampling':'NEAREST','downsampling':'NEAREST'}}]},
        'aggregation':{'timeRange':{'from':start,'to':end},'aggregationInterval':{'of':'P1D'},
                       'resx':c['resolution_m'],'resy':c['resolution_m'],'evalscript':evalscript(index)},
        'calculations':{'value':{}}}

def weather_request(c):
    """Врати Open-Meteo параметри за acquire, без мрежно барање.

    c ги дава датумите и weather_model. Локацијата е аритметичка средина
    на темињата од надворешниот прстен без повтореното завршно теме,
    не површински тежишен центар. Се бараат дневни UTC врнежи, дожд,
    снег, референтна ET0 и минимална температура.
    """
    ring = c['geometry']['coordinates'][0][:-1]
    return dict(latitude=sum(p[1] for p in ring)/len(ring),longitude=sum(p[0] for p in ring)/len(ring),
                start_date=c['start_date'],end_date=c['end_date'],timezone='UTC',models=c['weather_model'],
                daily='precipitation_sum,rain_sum,snowfall_sum,et0_fao_evapotranspiration,temperature_2m_min')

def dump(path, data):
    """Запиши JSON data во Path path за acquire и создај родителски директориуми.

    Постојната датотека се заменува; UTF-8 записот е вовлечен, а NaN/Inf
    не се дозволени. Нема повратна вредност; грешките при запис се пренесуваат.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def acquire(c, destination, weather_only=False):
    """Преземи и зачувај нов кеш според c; врати manifest со исходот.

    destination мора да е непостоечка патека, инаку AcquisitionError.
    CLI, тетратките и data.load_data го повикуваат експлицитно. Прави
    мрежни барања и создава сурови JSON, барања, CSV, снимка на c,
    геометрија и manifest.json со SHA-256 за претходно запишаните датотеки.
    retrieved_at_utc е моментот на преземање; времињата на снимање доаѓаат
    од каталогот. weather_only=True ги прескокнува сателитските барања.

    manifest.status е complete само без запишани errors. Очекуваните
    AcquisitionError/ValueError во метеоролошката и сателитската фаза се
    собираат во errors и дозволуваат делумен кеш; прескокнат сателит или
    нула прифатени NDMI денови исто дава incomplete. Грешки пред тие фази
    или при датотечен запис може да се пренесат без завршен manifest.
    Нема лажни набљудувања при неуспех и не се пресметува филтер.
    """
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
    """CLI преземање со --config, задолжителен нов --output и --weather-only по избор.

    --env-file вчитува тајни без надвладување на постојната околина.
    Го повикува acquire (мрежа и датотеки), печати краток JSON исход
    и завршува со код 2 ако status не е complete. argparse не чита docstrings.
    """
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
