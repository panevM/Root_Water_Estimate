# Adding another filter

Implement `ScalarFilter` in `root_zone_water/filters/scalar.py` (or a separate module):

```python
class MyFilter(ScalarFilter):
    def predict(self, transition, process_covariance, scale=1.0):
        # Propagate transition(W); process_covariance is course R_t in mm squared per day.
        # Set self.mean and self.variance; return algorithm diagnostics.
        ...
    def update(self, observation, measurement_covariance):
        # observation is proxy storage in mm; measurement_covariance is course Q_t in mm squared.
        # Return innovation_mm, S_mm2, gain and boundary-adjustment diagnostics.
        ...
```

Register in `FILTERS` and extend config validation. Keep state initialization and every forcing,
proxy conversion, quality decision and timing policy in the runner. Numerical imports must not
contact APIs. Tests should check an analytic affine case and covariance validity.

A particle filter could represent W with weighted particles, propagate with daily process noise,
weight with the same scalar proxy likelihood, normalize using log weights and resample only when
effective sample size drops below a declared threshold. Specify reproducible random seeds, bounded
state treatment and particle count; avoid silently changing the proxy or introducing simulated
ground truth. A PF implementation is intentionally outside this delivery.

The EKF uses the analytical `water_balance_jacobian` from `root_zone_water/model.py`. At switching
surfaces the derivative is not unique, so the implementation documents a one-sided branch
convention. UKF uses n=1, alpha=1, beta=2, kappa=0. Sigma points outside [0,Wsaturation] are
clipped and counted, which distorts moments and is only an approximation. The linear identity
measurement update uses the scalar course covariance correction for either predicted approximation.
Posterior means are bounded; any displacement is recorded and its square added to variance as
a modest conservative safeguard. This is not an exact truncated/constrained posterior. Gaussian
uncertainty bands may extend beyond physical bounds and do not represent validated coverage.
