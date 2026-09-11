"""One shared linear-observation heuristic. Never a field measurement."""
import math
import numpy as np

def from_ndmi(ndmi, spatial_std, p, c):
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
