"""Small analytic fixtures only; never published as observation data."""
import copy
import json
import socket
import numpy as np
import pandas as pd
import pytest
from root_zone_water.config import load_config, parameters, generate_config
from root_zone_water.model import budget, stress, calendar_kc, available_water, depletion, water_balance_jacobian
from root_zone_water.proxy import from_ndmi
from root_zone_water.filters import EKF, UKF
from root_zone_water.acquisition import interval, statistics_request, projected_geometry, SentinelClient, AcquisitionError
from root_zone_water.data import load_data, satellite_table, weather_table
from root_zone_water.runner import run, input_history

@pytest.fixture
def c(): return load_config()

def fixture_weather(c,n=3):
    c=c|{'start_date':'2025-05-01','end_date':f'2025-05-{n:02d}'}
    w=pd.DataFrame(dict(date=pd.date_range(c['start_date'],c['end_date']).strftime('%Y-%m-%d'),
                        precipitation_mm=0.,rain_mm=0.,snowfall_cm=0.,et0_mm=4.,temperature_min_c=10.))
    return c,w

def fixture_sat():
    return pd.DataFrame([dict(date='2025-05-01',ndvi=.7,ndmi=.2,ndmi_spatial_std=np.nan,
                             ndvi_accepted=True,ndmi_accepted=True,observation_id='unit-fixture',
                             acquisition_time_utc='2025-05-01T10:00:00Z',rejection_reason='')])

def test_generate_config_writes_dataset_selection(tmp_path):
    output = tmp_path / 'custom-config.json'
    config = generate_config('2025-06-01', '2025-06-30', 'ukf', 'data/student-set', output_path=output)
    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved == config
    assert saved['start_date'] == '2025-06-01'
    assert saved['end_date'] == '2025-06-30'
    assert saved['filter'] == 'ukf'
    assert saved['cache_dir'] == 'data/student-set'
    assert saved['mode'] == 'cached'

def test_root_scaling_stress_accounting(c):
    p,_,_=parameters(c); p2,_,_=parameters(c|{'root_depth_scale':2})
    assert p2.fc==2*p.fc and p2.wp==2*p.wp and p2.taw==2*p.taw
    assert stress(p.wp,p)==0 and stress(p.fc-p.raw,p)==pytest.approx(1)
    assert stress(p.fc,p)==1
    assert available_water(p.wp+.3*p.taw,p)+depletion(p.wp+.3*p.taw,p)==pytest.approx(p.taw)
    z=from_ndmi(.2,None,p,c); z2=from_ndmi(.2,None,p2,c)
    assert z2['proxy_mm']==2*z['proxy_mm'] and z2['measurement_covariance_Qt_mm2']==4*z['measurement_covariance_Qt_mm2']

@pytest.mark.parametrize('w_fraction',[0,.2,.6,1])
@pytest.mark.parametrize('rain',[0,30,2000])
def test_budget_conserves(c,w_fraction,rain):
    p,_,_=parameters(c)
    value,flux=budget(w_fraction*p.saturation,rain,1000,1.15,p)
    assert abs(flux['balance_error_mm'])<1e-10
    assert 0<=value<=p.saturation and flux['irrigation_mm']==0
    assert flux['eta_mm']>=0 and flux['drainage_mm']>=0 and flux['overflow_mm']>=0

def test_linear_case_and_course_covariance():
    for cls in (EKF,UKF):
        f=cls(10,4)
        if cls is EKF:
            f.predict(lambda x:.8*x+3,9,state_jacobian=.8)
        else:
            f.predict(lambda x:.8*x+3,9)
        assert f.mean==pytest.approx(11)
        assert f.variance==pytest.approx(.64*4+9)
        p=f.variance; info=f.update(12,16)
        assert f.mean==pytest.approx(11+p/(p+16))
        assert f.variance==pytest.approx(p*16/(p+16))
        assert info['innovation_mm']==pytest.approx(1)
        with pytest.raises(ValueError): f.update(12,0)
        with pytest.raises(ValueError): f.predict(lambda x:x,-1)

def test_water_balance_jacobian_smooth_branch(c):
    p,_,_=parameters(c)
    w = p.wp + .4*p.taw
    rain, et0, kc = 2., 3., 1.
    analytic = water_balance_jacobian(w,rain,et0,kc,p)
    step = 1e-4
    numeric = (budget(w+step,rain,et0,kc,p)[0]-budget(w-step,rain,et0,kc,p)[0])/(2*step)
    assert analytic == pytest.approx(numeric,abs=1e-8)

def test_water_balance_jacobian_threshold_convention(c):
    p,_,_=parameters(c)
    # Exact caps use the documented lower-storage branch convention.
    assert water_balance_jacobian(p.saturation,0.,4.,1.,p) == pytest.approx(.9)
    assert water_balance_jacobian(p.wp,0.,4.,1.,p) == pytest.approx(1.)

def test_daily_once_no_forward_fill_causal(c):
    c,w=fixture_weather(c); c['kc_mode']='ndvi'
    s=fixture_sat(); r=run(w,s,c)
    assert len(r)==3 and r['updated'].sum()==1
    assert r['kc_source'].tolist()==['calendar_fallback','last_valid_ndvi','last_valid_ndvi']
    assert r.loc[1:,'innovation_mm'].isna().all()
    assert np.isfinite(r['P_posterior_mm2']).all() and (r['P_posterior_mm2']>=0).all()
    assert r['irrigation_mm'].eq(0).all()
    pd.testing.assert_frame_equal(r,run(w,s,c))
    with pytest.raises(ValueError): run(w,pd.concat([s,s]),c)
    with pytest.raises(ValueError): run(pd.concat([w,w.iloc[:1]]),s,c)
    with pytest.raises(ValueError): run(w.iloc[1:],s,c)
    for mode in ('ekf','ukf','open_loop'):
        other=run(w,s,c,mode)
        assert other['kc'].tolist()==r['kc'].tolist()
        assert other.loc[0,'proxy_mm']==r.loc[0,'proxy_mm']

def test_ndvi_and_ndmi_independent(c):
    c,w=fixture_weather(c); s=fixture_sat()
    s['ndmi_accepted']=False; s['ndmi']=np.nan
    r=run(w,s,c|{'kc_mode':'ndvi'})
    assert r['updated'].sum()==0 and r.loc[1,'kc_source']=='last_valid_ndvi'
    s=fixture_sat(); s['ndvi_accepted']=False; s['ndvi']=np.nan
    assert run(w,s,c)['updated'].sum()==1
    s=fixture_sat(); c['ndvi_max_age_days']=0
    assert run(w,s,c|{'kc_mode':'ndvi'})['kc_source'].eq('calendar_fallback').all()

def test_proxy_censoring_and_missing_std(c):
    p,_,_=parameters(c)
    r=from_ndmi(.2,None,p,c)['measurement_covariance_Qt_mm2']
    assert r>0
    clipped=from_ndmi(-.9,0,p,c)
    assert clipped['endpoint_clipped'] and clipped['proxy_mm']==p.wp
    assert clipped['measurement_covariance_Qt_mm2']>=r
    with pytest.raises(ValueError): from_ndmi(np.nan,None,p,c)

def test_calendar_boundaries(c):
    _,crop,_=parameters(c)
    assert calendar_kc('2025-04-01',c['planting_date'],crop)==(.3,'initial')
    assert calendar_kc('2025-05-21',c['planting_date'],crop)==(1.15,'mid')
    with pytest.raises(ValueError): calendar_kc('2025-01-01',c['planting_date'],crop)

def test_request_metric_polygon_full_end(c):
    body=statistics_request(c,'ndmi')
    assert interval(c)[1]=='2025-07-30T00:00:00Z'
    assert body['aggregation']['resx']==body['aggregation']['resy']==20
    assert 'bbox' not in body['input']['bounds']
    ring=np.array(projected_geometry(c)['coordinates'][0])
    assert 800 < np.ptp(ring[:,0]) < 900 and 1100 < np.ptp(ring[:,1]) < 1200
    assert '[0,1,3,8,9,10,11]' in body['aggregation']['evalscript']
    assert 's.B8A-s.B11' in body['aggregation']['evalscript']

def test_cloud_rejection_catalog_duplicate(c):
    c=c|{'start_date':'2025-05-01','end_date':'2025-05-01'}
    item={'interval':{'from':'2025-05-01T00:00:00Z','to':'2025-05-02T00:00:00Z'},
          'outputs':{'value':{'bands':{'B0':{'stats':{'mean':.2,'stDev':.1,'sampleCount':2400,'noDataCount':2300}}}}}}
    raws={k:{'data':[copy.deepcopy(item)]} for k in ('ndvi','ndmi','vegetation')}
    f={'id':'unit-fixture','properties':{'datetime':'2025-05-01T10:00:00Z'}}
    r=satellite_table(raws,[f],c)
    assert not r.loc[0,'ndmi_accepted']
    with pytest.raises(ValueError): satellite_table(raws,[f,f],c)
    raws['ndmi']['data'].append(item)
    with pytest.raises(ValueError): satellite_table(raws,[f],c)

def test_grid_counts_and_acquisition_ambiguity(c):
    c=c|{'start_date':'2025-05-01','end_date':'2025-05-01'}
    item={'interval':{'from':'2025-05-01T00:00:00.000Z','to':'2025-05-02T00:00:00.000Z'},
          'outputs':{'value':{'bands':{'B0':{'stats':{'mean':.2,'stDev':.1,'sampleCount':2400,'noDataCount':10}}}}}}
    raws={k:{'data':[copy.deepcopy(item)]} for k in ('ndvi','ndmi','vegetation')}
    f={'id':'unit-fixture','properties':{'datetime':'2025-05-01T10:00:00Z'}}
    assert satellite_table(raws,[f],c).loc[0,'ndmi_accepted']
    second={'id':'unit-fixture-2','properties':{'datetime':'2025-05-01T11:00:00Z'}}
    assert not satellite_table(raws,[f,second],c).loc[0,'ndmi_accepted']
    raws['ndmi']['data'][0]['outputs']['value']['bands']['B0']['stats']['sampleCount']=10000000
    with pytest.raises(ValueError,match='metric grid'): satellite_table(raws,[f],c)

def test_ukf_bounds_and_water_adjustments(c):
    p,_,_=parameters(c)
    f=UKF(p.wp,4*p.saturation**2,(0,p.saturation))
    info=f.predict(lambda x:budget(x,0,4,1.15,p)[0],9,p.taw)
    assert info['sigma_points_clipped']==2 and np.isfinite(f.variance) and f.variance>=0
    f.update(10000,1)
    assert 0<=f.mean<=p.saturation and f.variance>=0
    c,w=fixture_weather(c)
    result=run(w,fixture_sat(),c,'ukf')
    np.testing.assert_allclose(result['W_posterior_mm'],result['physical_prediction_mm']+
                               result['distribution_prediction_adjustment_mm']+result['assimilation_adjustment_mm'])

def test_offline_cache_and_repeatability(c,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('Offline mode attempted network')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    w,s,m=load_data(c)
    r=run(w,s,c)
    assert len(r)==len(w)==90
    assert r['irrigation_mm'].eq(0).all()

def test_token_reuse(monkeypatch):
    monkeypatch.setenv('SENTINEL_CLIENT_ID','unit-fixture')
    monkeypatch.setenv('SENTINEL_CLIENT_SECRET','unit-fixture')
    calls=[]
    def fake(*a,**k): calls.append(1); return {'access_token':'fixture','expires_in':300}
    monkeypatch.setattr('root_zone_water.acquisition.request_json',fake)
    client=SentinelClient()
    assert client.headers()==client.headers() and len(calls)==1

def test_missing_weather_snow_and_timestamp(c):
    c,w=fixture_weather(c); s=fixture_sat()
    for col,value in [('precipitation_mm',np.nan),('snowfall_cm',1),('temperature_min_c',-1)]:
        broken=w.copy(); broken.loc[0,col]=value
        with pytest.raises(ValueError): run(broken,s,c)
    s.loc[0,'acquisition_time_utc']='2025-05-02T10:00:00Z'
    with pytest.raises(ValueError): run(w,s,c)
