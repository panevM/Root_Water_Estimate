"""Поврзување на влезовите, водниот биланс и филтрите во дневен UTC тек.

Почнете со run за избор и иницијализација на филтерот, input_history за
временското усогласување и proxy.from_ndmi за набљудувањето во mm.
FILTERS е регистарот во filters/__init__.py; математиката е во filters/scalar.py
и model.py. compare и sensitivity повторуваат пресметки врз исти влезови,
а main чита податоци и запишува CSV резултати.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_config, parameters
from .data import load_data, validate_weather
from .filters import FILTERS
from .model import budget, calendar_kc, ndvi_kc, finite_difference_derivative, water_balance_jacobian
from .proxy import from_ndmi

def input_history(weather, satellite, c):
    """Подготви дневни влезови за run, без промена на влезните DataFrame табели.

    weather мора да ги покрива сите датуми YYYY-MM-DD од c со валидно време
    без снег/мраз; satellite има најмногу еден ред по ден со независни
    ndvi_accepted и ndmi_accepted ознаки. Дупликати на ден или идентитет
    на продукт, набљудување надвор од периодот и невалидно време на прифатена
    снимка предизвикуваат ValueError. Времето мора да има временска зона
    и да припаѓа во дневниот UTC интервал [почеток, следна полноќ).

    Враќа листа речници со метеоролошките полиња, kc, stage, kc_source,
    ndvi_input_age_days, observation и границите interval_*_utc во ISO формат.
    Kc прво доаѓа од календарот; во режим ndvi може да го замени последниот
    прифатен NDVI со возраст од 0 до ndvi_max_age_days на почетокот на денот.
    Тековниот NDVI се памети дури по составување на денешниот ред, па влијае
    најрано следниот ден. Стар/отсутен NDVI значи calendar_fallback.
    NDMI не се пренесува во следни денови: observation е празен ако нема ред.
    """
    weather = validate_weather(weather.copy(),c)
    sat = satellite.copy().sort_values('date')
    if sat['date'].duplicated().any(): raise ValueError('Duplicate satellite day; composite before running')
    if not set(sat['date']).issubset(set(weather['date'])): raise ValueError('Observation outside weather interval')
    ids = sat.loc[sat['observation_id'].fillna('') != '', 'observation_id']
    products = [pid for group in ids for pid in group.split(';')]
    if len(products) != len(set(products)): raise ValueError('Repeated observation/product identity')
    by_date = sat.set_index('date').to_dict('index')
    p,crop,_ = parameters(c)
    history, last_ndvi, last_time = [], None, None
    for row in weather.to_dict('records'):
        day = row['date']
        start = pd.Timestamp(day,tz='UTC')
        kc,stage = calendar_kc(day,c['planting_date'],crop)
        source,ndvi_age = 'calendar', np.nan
        if last_time is not None: ndvi_age = (start-last_time).total_seconds()/86400
        if c['kc_mode']=='ndvi' and last_ndvi is not None and 0 <= ndvi_age <= c['ndvi_max_age_days']:
            kc,source = ndvi_kc(last_ndvi,crop),'last_valid_ndvi'
        elif c['kc_mode']=='ndvi': source='calendar_fallback'
        observation = by_date.get(day,{})
        row.update(kc=kc,stage=stage,kc_source=source,ndvi_input_age_days=ndvi_age,
                   observation=observation,interval_start_utc=start.isoformat(),
                   interval_end_utc=(start+pd.Timedelta(days=1)).isoformat())
        history.append(row)
        if observation:
            ts = observation.get('acquisition_time_utc')
            any_accepted = observation.get('ndvi_accepted',False) or observation.get('ndmi_accepted',False)
            if any_accepted:
                if not ts: raise ValueError('Accepted observation lacks actual acquisition time')
                actual = pd.Timestamp(ts)
                if actual.tzinfo is None or not start <= actual < start+pd.Timedelta(days=1):
                    raise ValueError('Acquisition time does not match its UTC daily interval')
            if observation.get('ndvi_accepted',False):
                value = observation.get('ndvi')
                if not np.isfinite(value) or not -1 <= value <= 1: raise ValueError('Invalid accepted NDVI')
                last_ndvi,last_time = value,actual
    return history

def run(weather,satellite,c,filter_name=None):
    """Изврши дневна процена и врати DataFrame со состојби и дијагностика.

    weather и satellite се табелите од data.load_data; c е проверена
    конфигурација. filter_name го надвладува c['filter']; FILTERS го избира
    класното име (непознат клуч крева KeyError). Еден објект f се создава
    пред циклусот со μ_0=wp+initial_available_fraction·TAW,
    Σ_0=(initial_std_taw_fraction·TAW)² и граници [0,заситеност].
    Истиот f ја задржува процената од претходниот ден.

    Секој ден: земи ја старата процена, пресметај физички биланс и R_t,
    повикај predict, па најмногу еднаш update со прифатен NDMI претворен
    во mm. Прифатена снимка во текот на денот се користи за корекција на
    крајот на денот; тоа е временска апроксимација, без поддневен премин.
    Без прифатен NDMI има само предвидување, без измислено набљудување.
    За open_loop се пресметува индиректната процена и иновацијата кога
    се достапни, но update никогаш не се повикува и updated останува False.

    Коваријансата на шумот на процесот е
    R_t=(process_std_taw_fraction·TAW)², во mm², за отстапувања на моделот.
    Со weather_uncertainty се додаваат (∂g/∂rain)²·rain_std_mm² и
    (∂g/∂ET0)²·et0_std_mm², без вкрстена коваријанса. Овие изводи се
    нумерички, при фиксно before, и се mm/mm. rain_std_mm и et0_std_mm
    се неизвесности во mm, а rain и et0 се самите дневни влезни вредности.
    Тука нема посебен пренос на неизвесноста на Kc или на почвените параметри.

    W_prior_mm/P_prior_mm2 се μ̄_t/Σ̄_t, а W_posterior_mm/P_posterior_mm2
    се μ_t/Σ_t по можната корекција. start_storage_mm/start_P_mm2 се
    претходните вредности. predicted_proxy_mm=μ̄_t бидејќи h(W)=W.
    process_covariance_Rt_mm2 и measurement_covariance_Qt_mm2 се R_t и Q_t;
    innovation_mm, S_mm2 и gain ги следат равенките во ScalarFilter.update.
    Без набљудување недостапните дијагностики се NaN.
    distribution_prediction_adjustment_mm е μ̄_t−g(before), што кај UKF
    ја вклучува разликата од преносот на распределбата; assimilation_adjustment_mm
    е μ_t−μ̄_t, вклучувајќи евентуална проекција по корекција. Овие промени
    не се физички водни текови. flux полињата се опишани во model.budget.
    Функцијата чита локални референтни табели преку parameters, но не
    презема податоци, не запишува резултати и не ги менува влезните табели.
    """
    p,_,_ = parameters(c)
    name = filter_name or c['filter']
    f = FILTERS[name](p.wp+c['initial_available_fraction']*p.taw,
                      (c['initial_std_taw_fraction']*p.taw)**2,(0,p.saturation))
    records = []
    for row in input_history(weather,satellite,c):
        before,before_p = f.result()
        # Вкупните врнежи во проверениот период без снег ги вклучуваат и пороите.
        rain,et0,kc = row['precipitation_mm'],row['et0_mm'],row['kc']
        transition = lambda w: budget(w,rain,et0,kc,p)[0]
        physical,flux = budget(before,rain,et0,kc,p)
        # R_t: коваријанса на шумот на процесот, со опционална неизвесност на времето.
        process_covariance = (c['process_std_taw_fraction']*p.taw)**2
        df_dp = df_det = 0.0
        if c['weather_uncertainty']:
            df_dp = finite_difference_derivative(lambda v:budget(before,v,et0,kc,p)[0],rain,p.taw,0)
            df_det = finite_difference_derivative(lambda v:budget(before,rain,v,kc,p)[0],et0,p.taw,0)
            process_covariance += df_dp**2*c['rain_std_mm']**2 + df_det**2*c['et0_std_mm']**2
        # По predict, објектот ги чува μ̄_t и Σ̄_t за тековниот ден.
        prediction = (f.predict(transition,process_covariance,p.taw,
                                state_jacobian=water_balance_jacobian(before,rain,et0,kc,p))
                      if name in ('ekf','open_loop') else
                      f.predict(transition,process_covariance,p.taw))
        prior,prior_p = f.result()
        obs = row['observation']
        detail = dict(proxy_mm=np.nan,measurement_covariance_Qt_mm2=np.nan,endpoint_clipped=False,innovation_mm=np.nan,
                      S_mm2=np.nan,gain=np.nan,posterior_bound_adjustment_mm=0.0,observation_age_days=np.nan)
        usable = bool(obs.get('ndmi_accepted',False))
        updated = usable and name != 'open_loop'
        if usable:
            # Q_t е коваријанса на шумот на мерењето за индиректната процена во mm.
            detail.update(from_ndmi(obs['ndmi'],obs.get('ndmi_spatial_std'),p,c))
            detail['observation_age_days'] = (pd.Timestamp(row['interval_end_utc'])-pd.Timestamp(obs['acquisition_time_utc'])).total_seconds()/86400
            detail.update(innovation_mm=detail['proxy_mm']-prior,S_mm2=prior_p+detail['measurement_covariance_Qt_mm2'])
            if updated: detail.update(f.update(detail['proxy_mm'],detail['measurement_covariance_Qt_mm2']))
        posterior,posterior_p = f.result()
        records.append({k:v for k,v in row.items() if k != 'observation'} | flux | prediction | detail |
                       dict(filter=name,start_storage_mm=before,start_P_mm2=before_p,W_prior_mm=prior,P_prior_mm2=prior_p,
                            predicted_proxy_mm=prior,W_posterior_mm=posterior,P_posterior_mm2=posterior_p,
                            process_covariance_Rt_mm2=process_covariance,
                            df_dprecipitation=df_dp,df_det0=df_det,ndmi_accepted=usable,updated=updated,
                            rejected=bool(obs) and not usable,rejection_reason=obs.get('rejection_reason','no_observation'),
                            observation_id=obs.get('observation_id',''),
                            assimilation_adjustment_mm=posterior-prior,
                            distribution_prediction_adjustment_mm=prior-physical))
    return pd.DataFrame(records)

def compare(weather,satellite,c):
    """Врати речник име→DataFrame за open_loop, ekf и ukf со исти влезови.

    main и тетратките го повикуваат со табели од load_data и конфигурација c.
    Секој run создава сопствен филтер; овој список не се проширува автоматски
    при додавање класа во FILTERS. Не запишува датотеки.
    """
    return {name:run(weather,satellite,c,name) for name in ('open_loop','ekf','ukf')}

def sensitivity(weather,satellite,c):
    """Врати речник сценарио→EKF резултат за истите табели weather и satellite.

    Тетратките ги прикажуваат сценаријата преку plotting; c се копира со
    измена на по една претпоставка. Постојната ознака 'Q variance x4'
    всушност ја зголемува основната коваријанса на шумот на процесот R_t
    четирипати, без метеоролошкиот додаток. 'R larger' ја удвојува само
    стандардната девијација на врската NDMI→W, дел од коваријансата на
    шумот на мерењето Q_t. Ознаките ја користат обратната нотација од курсот.
    Резултатите покажуваат чувствителност на претпоставки, не точност спрема
    теренски мерења. Влезните податоци и c не се менуваат.
    """
    changes = {'baseline':{},'Q variance x4':{'process_std_taw_fraction':c['process_std_taw_fraction']*2},
               'R larger':{'proxy_model_std_taw_fraction':c['proxy_model_std_taw_fraction']*2},
               'initial mean wetter':{'initial_available_fraction':.75},
               'initial variance x4':{'initial_std_taw_fraction':c['initial_std_taw_fraction']*2},
               'older proxy endpoints':{'m_dry':-.20,'m_wet':.60},
               'root depth x0.75':{'root_depth_scale':.75}}
    return {label:run(weather,satellite,c|change,'ekf') for label,change in changes.items()}

def diagnostics(results):
    """Сумирај речник име→резултат од run во DataFrame за main и тетратките.

    proxy_pairs брои достапни иновации, corrections вистински корекции,
    prior_proxy_MAE_mm/RMSE_mm се отстапувања од индиректната процена
    во mm, а max_physical_balance_error_mm е најголем апсолутен остаток
    на физичкиот биланс. Првите две грешки се NaN без парови и не се
    валидација со теренска вистина; парови може да има и за open_loop.
    """
    rows = []
    for name,result in results.items():
        innovations = result['innovation_mm'].dropna()
        rows.append(dict(filter=name,proxy_pairs=len(innovations),
                         prior_proxy_MAE_mm=innovations.abs().mean(),
                         prior_proxy_RMSE_mm=np.sqrt((innovations**2).mean()),
                         corrections=int(result['updated'].sum()),
                         max_physical_balance_error_mm=result['balance_error_mm'].abs().max()))
    return pd.DataFrame(rows)

def main():
    """CLI влез: читај --config и запиши споредба во --output (стандардно outputs).

    Создава директориум и запишува/заменува CSV за трите филтри и diagnostics,
    па печати резиме. load_data го следи режимот од конфигурацијата;
    main не задава destination, па режим api таму крева ValueError.
    Не користи docstring за argparse помош.
    """
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output',default='outputs')
    args=parser.parse_args()
    c=load_config(args.config)
    weather,satellite,manifest=load_data(c)
    results=compare(weather,satellite,c)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for name,result in results.items(): result.to_csv(out/f'{name}.csv',index=False)
    summary=diagnostics(results)
    summary.to_csv(out/'diagnostics.csv',index=False)
    print('Dataset status:',manifest['status'],'| No measured root-zone ground truth')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
