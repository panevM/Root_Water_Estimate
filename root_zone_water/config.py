"""Читање и проверка на конфигурацијата пред преземање или проценување.

load_config ги проверува поставките; parameters ги поврзува референтните
табели со model.Parameters. Почетната и процесната стандардна девијација
се задаваат како дел од TAW во initial_std_taw_fraction/process_std_taw_fraction.
rain_std_mm и et0_std_mm се метеоролошки неизвесности; ndmi_obs_std,
proxy_model_std_taw_fraction и clipped_R_multiplier го одредуваат Q_t
во proxy.from_ndmi. FILTERS во filters/__init__.py ги регистрира класите,
а дозволените имиња се проверуваат и во овој модул.
"""
import csv
import json
import math
from pathlib import Path
from .model import Parameters

ROOT = Path(__file__).resolve().parents[1]

def generate_config(start_date, end_date, filter_name, dataset_path,
                    output_path=None, base_path=None, planting_date=None):
    """Состави конфигурација за кориснички податоци, како во тетратките за преземање.

    start_date/end_date се вклучени датуми YYYY-MM-DD; filter_name е ekf,
    ukf или open_loop. dataset_path е непразна патека до кешот.
    base_path избира основен JSON (стандардно ROOT/config.json), а
    planting_date по избор го заменува датумот на садење. Враќа речник
    со mode='cached'; не ја прави целосната проверка од load_config.
    Ако има output_path, создава родителски директориуми и запишува/заменува
    JSON таму. Не менува основна датотека освен ако е избрана како излез.
    Невалидни датуми, обратен период, име на филтер или празна патека
    предизвикуваат ValueError; грешки при читање/пишување се пренесуваат.
    """
    from datetime import date
    allowed = ('ekf', 'ukf', 'open_loop')
    start = date.fromisoformat(str(start_date))
    end = date.fromisoformat(str(end_date))
    if end < start:
        raise ValueError('end_date must not precede start_date')
    if filter_name not in allowed:
        raise ValueError(f'filter_name must be one of {allowed}')
    if not str(dataset_path).strip():
        raise ValueError('dataset_path must not be empty')
    source = Path(base_path or ROOT / 'config.json')
    config = json.loads(source.read_text(encoding='utf-8'))
    plant = date.fromisoformat(str(planting_date or config['planting_date']))
    config.update(start_date=start.isoformat(), end_date=end.isoformat(),
                  planting_date=plant.isoformat(), filter=filter_name, 
                  mode='cached', cache_dir=str(dataset_path))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return config

def load_config(path=None):
    """Прочитај JSON од path или ROOT/config.json и врати проверен речник.

    Се користи од CLI, тетратките и тестовите. Бара UTC, assumed_rainfed,
    мрежа од 20 m, подредени NDMI крајни точки во [-1,1] и дозволени
    режими. Почетниот достапен дел е [0,1]; зададените стандардни девијации
    како делови од TAW и ndmi_obs_std се позитивни. Метеоролошките
    стандардни девијации (mm) и максималната старост на NDVI (денови)
    може да се нула. min_valid_fraction е (0,1], min_valid_pixels≥1,
    а clipped_R_multiplier≥1 го зголемува Q_t при крајните точки.
    Крева ValueError за прекршени проверки; недостасувачки клучеви или
    невалиден JSON не се пополнуваат автоматски. Само чита датотека.
    """
    c = json.loads(Path(path or ROOT / 'config.json').read_text())
    if c['irrigation_mode'] != 'assumed_rainfed' or c['timezone'] != 'UTC':
        raise ValueError('Teaching runner requires assumed_rainfed and UTC')
    if c['kc_mode'] not in ('calendar', 'ndvi') or c['filter'] not in ('ekf', 'ukf', 'open_loop'):
        raise ValueError('Unknown Kc mode or filter')
    if not -1 <= c['m_dry'] < c['m_wet'] <= 1:
        raise ValueError('Proxy endpoints must be ordered within [-1,1]')
    if not 0 <= c['initial_available_fraction'] <= 1:
        raise ValueError('Initial fraction must be in [0,1]')
    for name in ('initial_std_taw_fraction', 'process_std_taw_fraction',
                 'proxy_model_std_taw_fraction', 'ndmi_obs_std'):
        if not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not 0 <= c['precipitation_effectiveness'] <= 1 or not 0 < c['min_valid_fraction'] <= 1:
        raise ValueError('Invalid effectiveness/coverage')
    if c['clipped_R_multiplier'] < 1 or c['min_valid_pixels'] < 1:
        raise ValueError('Invalid quality safeguards')
    for name in ('rain_std_mm','et0_std_mm','ndvi_max_age_days'):
        if not math.isfinite(c[name]) or c[name] < 0: raise ValueError('Invalid '+name)
    if c['resolution_m'] != 20: raise ValueError('This processing version requires a common 20 m grid')
    from datetime import date
    if date.fromisoformat(c['end_date']) < date.fromisoformat(c['start_date']):
        raise ValueError('Study end precedes start')
    return c

def reference(table, key, value):
    """Врати го првиот CSV ред од references/table со row[key]==value.

    parameters ги бара културата и почвата; полињата остануваат стрингови.
    Само чита локална датотека; без совпаѓање се пренесува StopIteration.
    """
    with (ROOT / 'references' / table).open(newline='') as f:
        return next(row for row in csv.DictReader(f) if row[key] == value)

def parameters(c):
    """Врати (Parameters, ред за култура, ред за почва) од конфигурацијата c.

    runner и acquire ги користат локалните crops.csv/soils.csv. Длабочината
    во m се множи со root_depth_scale и останува фиксна низ деновите;
    theta_* се во m³/m³, а p, drainage_fraction и effectiveness се делови.
    Parameters ги проверува физичките вредности. Само чита референтни
    табели; не ги менува c или табелите.
    """
    crop = reference('crops.csv', 'crop', c['crop'])
    soil = reference('soils.csv', 'soil', c['soil'])
    p = Parameters(float(crop['root_depth_m']) * c['root_depth_scale'],
                   float(soil['theta_fc_m3m3']), float(soil['theta_wp_m3m3']),
                   float(soil['theta_sat_m3m3']), float(crop['p']),
                   float(soil['drainage_fraction_day']), c['precipitation_effectiveness'])
    return p, crop, soil
