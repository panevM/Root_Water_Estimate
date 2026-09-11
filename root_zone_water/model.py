"""Daily root-zone water balance; the state is total storage in millimetres."""
from dataclasses import dataclass
from datetime import date
import math
import numpy as np

@dataclass(frozen=True)
class Parameters:
    root_depth_m: float
    theta_fc: float
    theta_wp: float
    theta_sat: float
    p: float
    drainage_fraction: float
    effectiveness: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError('Nonfinite parameter')
        if not (self.root_depth_m > 0 and 0 < self.theta_wp < self.theta_fc < self.theta_sat <= 1
                and 0 < self.p < 1 and 0 <= self.drainage_fraction <= 1 and 0 <= self.effectiveness <= 1):
            raise ValueError('Invalid soil/crop parameters')

    @property
    def wp(self): return 1000 * self.root_depth_m * self.theta_wp
    @property
    def fc(self): return 1000 * self.root_depth_m * self.theta_fc
    @property
    def saturation(self): return 1000 * self.root_depth_m * self.theta_sat
    @property
    def taw(self): return self.fc - self.wp
    @property
    def raw(self): return self.p * self.taw

def available_water(w, p): return float(np.clip(w - p.wp, 0, p.taw))
def depletion(w, p): return float(np.clip(p.fc - w, 0, p.taw))
def stress(w, p): return float(np.clip((w - p.wp) / ((1 - p.p) * p.taw), 0, 1))

def budget(w, rain_mm, et0_mm, kc, p):
    if not all(math.isfinite(v) for v in (w, rain_mm, et0_mm, kc)):
        raise ValueError('Nonfinite state/forcing')
    if not 0 <= w <= p.saturation or min(rain_mm, et0_mm, kc) < 0:
        raise ValueError('State/forcing outside physical domain')
    effective = p.effectiveness * rain_mm
    wet = w + effective
    overflow = max(wet - p.saturation, 0.0)
    wet -= overflow
    ks = stress(wet, p)
    eta = min(ks * kc * et0_mm, max(wet - p.wp, 0.0))
    after_et = wet - eta
    drainage = p.drainage_fraction * max(after_et - p.fc, 0.0)
    result = after_et - drainage
    flux = dict(rain_mm=rain_mm, effective_rain_mm=effective, irrigation_mm=0.0,
                ineffective_rain_mm=rain_mm-effective, overflow_mm=overflow,
                eta_mm=eta, drainage_mm=drainage, ks=ks, wet_storage_mm=wet,
                physical_prediction_mm=result)
    flux['balance_error_mm'] = result - (w + rain_mm - flux['ineffective_rain_mm'] - overflow - eta - drainage)
    return result, flux

def water_balance_jacobian(w, rain_mm, et0_mm, kc, p):
    """Analytical d(budget)/d(storage) using the active piecewise branches.

    At exact thresholds the convention is to use the lower-storage branch for
    overflow, stress clipping and drainage; the capped-water branch is used
    when the evapotranspiration minimum is tied.  These are one-sided branch
    conventions, not claims of a unique classical derivative at a kink.
    """
    if not all(math.isfinite(v) for v in (w, rain_mm, et0_mm, kc)):
        raise ValueError('Nonfinite state/forcing')
    if not 0 <= w <= p.saturation or min(rain_mm, et0_mm, kc) < 0:
        raise ValueError('State/forcing outside physical domain')

    wet_before_overflow = w + p.effectiveness * rain_mm
    if wet_before_overflow <= p.saturation:
        wet = wet_before_overflow
        d_wet = 1.0
    else:
        wet = p.saturation
        d_wet = 0.0

    stress_denominator = (1 - p.p) * p.taw
    stress_argument = (wet - p.wp) / stress_denominator
    if stress_argument <= 0:
        d_stress = 0.0
        stress_value = 0.0
    elif stress_argument >= 1:
        d_stress = 0.0
        stress_value = 1.0
    else:
        d_stress = d_wet / stress_denominator
        stress_value = stress_argument

    available = max(wet - p.wp, 0.0)
    d_available = d_wet if wet > p.wp else 0.0
    demand = stress_value * kc * et0_mm
    d_demand = d_stress * kc * et0_mm
    if demand < available:
        d_eta = d_demand
    else:
        d_eta = d_available

    d_after_et = d_wet - d_eta
    after_et = wet - min(demand, available)
    if after_et > p.fc:
        d_drainage = p.drainage_fraction * d_after_et
    else:
        d_drainage = 0.0
    return d_after_et - d_drainage

def finite_difference_derivative(f, x, scale=1.0, lower=-math.inf, upper=math.inf):
    """Numerical derivative reserved for weather-uncertainty propagation."""
    step = np.cbrt(np.finfo(float).eps) * max(abs(x), abs(scale), 1.0)
    lo, hi = max(lower, x-step), min(upper, x+step)
    if hi <= lo: raise ValueError('No finite-difference interval')
    return (f(hi)-f(lo))/(hi-lo)

def calendar_kc(day, planting_date, crop):
    age = (date.fromisoformat(str(day)) - date.fromisoformat(planting_date)).days
    a,b,c,d = [int(crop[k+'_days']) for k in ('initial','development','mid','late')]
    ki,km,ke = [float(crop['kc_'+k]) for k in ('initial','mid','end')]
    if age < 0 or age >= a+b+c+d: raise ValueError('Date outside assumed crop calendar')
    if age < a: return ki, 'initial'
    if age < a+b: return ki+(km-ki)*(age-a)/b, 'development'
    if age < a+b+c: return km, 'mid'
    return km+(ke-km)*(age-a-b-c)/d, 'late'

def ndvi_kc(ndvi, crop):
    ki,km,ke = [float(crop['kc_'+k]) for k in ('initial','mid','end')]
    # The optional NDVI mode uses the configured wheat intercept of 0.40.
    if crop['crop'] == 'wheat': ki = 0.40
    slope = (km-ki)/(max(float(crop['default_ndvi']), .25)-.20)
    return float(np.clip(ki+slope*(ndvi-.20), min(ki,ke), max(ki,km,ke)))
