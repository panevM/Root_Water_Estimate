"""Претворање NDMI во набљудување за заедничката корекција на EKF и UKF.

runner.run добива индиректна процена на количеството вода добиена од NDMI,
со коваријанса на шумот на мерењето Q_t. Таа е заснована на претпоставени
крајни точки и почвени параметри, без калибрација со теренски мерења.
Потоа филтерот користи h(W)=W во mm; не го набљудува суровиот NDMI.
"""
import math
import numpy as np

def from_ndmi(ndmi, spatial_std, p, c):
    """Врати индиректна процена од прифатен NDMI и нејзината неизвесност.

    runner.run ги предава ndmi (конечен индекс во [-1,1]), spatial_std
    (просторна стандардна девијација на индексот или None/NaN), Parameters p
    и конфигурацијата c. m_dry е претпоставената сува крајна точка што
    одговара на wp, а m_wet е влажната што одговара на fc, не на заситеноста.
    fraction=(ndmi−m_dry)/(m_wet−m_dry) се ограничува во [0,1], па
    proxy_mm=wp+TAW·fraction лежи во [wp,fc]. endpoint_clipped е True и
    точно на крајните точки, иако таму нема нумеричко поместување.

    ndmi_obs_std е претпоставена стандардна девијација во единици на индекс.
    Ако use_spatial_std_heuristic е вклучено и spatial_std е конечна,
    се зема поголемата од нив. Просторната варијабилност меѓу пиксели
    не е стандардна грешка на средината; тука нема делење со √број_пиксели.
    sigma_index_water_mm=TAW/(m_wet−m_dry)·sigma_index ја пренесува
    неизвесноста на индексот во mm. Посебно, sigma_proxy_model_mm=
    proxy_model_std_taw_fraction·TAW ја претставува претпоставената
    неизвесност во самата врска NDMI→W, а не разликите меѓу пикселите.

    measurement_covariance_Qt_mm2 е збирот на квадратите на овие две
    стандардни девијации во mm², без член за нивна меѓусебна коваријанса.
    Ако endpoint_clipped е True, целиот збир се множи со
    clipped_R_multiplier: и покрај името R, оваа поставка го менува Q_t.
    Наклонот TAW/width се користи и при отсекување; ова е хеуристичко
    зголемување на неизвесноста, не точен модел на отсечено набљудување.
    Поголемо Q_t намалува доверба, но не корегира експлицитно систематска
    пристрасност на врската NDMI→W.

    Ги враќа сите пет именувани полиња погоре, без промена на p или c.
    Крева ValueError за невалиден NDMI, неуредени крајни точки или негативна
    конечна spatial_std кога хеуристиката е вклучена. Останатите поставки
    се очекува претходно да ги провери config.load_config.
    """
    if not math.isfinite(ndmi) or not -1 <= ndmi <= 1:
        raise ValueError('Invalid NDMI')
    width = c['m_wet'] - c['m_dry']
    if width <= 0: raise ValueError('Unordered endpoints')
    fraction = (ndmi-c['m_dry'])/width
    clipped = not 0 < fraction < 1
    sigma_index = c['ndmi_obs_std']
    if c['use_spatial_std_heuristic'] and spatial_std is not None and math.isfinite(spatial_std):
        if spatial_std < 0: raise ValueError('Negative spatial dispersion')
        sigma_index = max(sigma_index, spatial_std)
    sigma_w = p.taw / width * sigma_index
    sigma_model = c['proxy_model_std_taw_fraction'] * p.taw
    r = (sigma_w**2 + sigma_model**2) * (c['clipped_R_multiplier'] if clipped else 1)
    return dict(proxy_mm=p.wp+p.taw*float(np.clip(fraction,0,1)), measurement_covariance_Qt_mm2=r,
                endpoint_clipped=clipped, sigma_index_water_mm=sigma_w,
                sigma_proxy_model_mm=sigma_model)
