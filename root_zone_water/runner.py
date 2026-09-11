"""One daily UTC prediction, then at most one sparse scalar correction."""
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
    p,_,_ = parameters(c)
    name = filter_name or c['filter']
    f = FILTERS[name](p.wp+c['initial_available_fraction']*p.taw,
                      (c['initial_std_taw_fraction']*p.taw)**2,(0,p.saturation))
    records = []
    for row in input_history(weather,satellite,c):
        before,before_p = f.result()
        # In a validated snow-free interval total precipitation includes liquid showers too.
        rain,et0,kc = row['precipitation_mm'],row['et0_mm'],row['kc']
        transition = lambda w: budget(w,rain,et0,kc,p)[0]
        physical,flux = budget(before,rain,et0,kc,p)
        # Process covariance R_t: model discrepancy plus optional weather uncertainty.
        process_covariance = (c['process_std_taw_fraction']*p.taw)**2
        df_dp = df_det = 0.0
        if c['weather_uncertainty']:
            df_dp = finite_difference_derivative(lambda v:budget(before,v,et0,kc,p)[0],rain,p.taw,0)
            df_det = finite_difference_derivative(lambda v:budget(before,rain,v,kc,p)[0],et0,p.taw,0)
            process_covariance += df_dp**2*c['rain_std_mm']**2 + df_det**2*c['et0_std_mm']**2
        # Prediction: the filter now has mu_bar_t and Sigma_bar_t.
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
            # Measurement covariance Q_t belongs to the speculative proxy, not raw NDMI.
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
    return {name:run(weather,satellite,c,name) for name in ('open_loop','ekf','ukf')}

def sensitivity(weather,satellite,c):
    changes = {'baseline':{},'Q variance x4':{'process_std_taw_fraction':c['process_std_taw_fraction']*2},
               'R larger':{'proxy_model_std_taw_fraction':c['proxy_model_std_taw_fraction']*2},
               'initial mean wetter':{'initial_available_fraction':.75},
               'initial variance x4':{'initial_std_taw_fraction':c['initial_std_taw_fraction']*2},
               'older proxy endpoints':{'m_dry':-.20,'m_wet':.60},
               'root depth x0.75':{'root_depth_scale':.75}}
    return {label:run(weather,satellite,c|change,'ekf') for label,change in changes.items()}

def diagnostics(results):
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
