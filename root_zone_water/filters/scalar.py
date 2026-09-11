"""Small scalar EKF, UKF and prediction-only filter for the teaching model."""
from abc import ABC, abstractmethod
import math
import numpy as np

class ScalarFilter(ABC):
    """Minimal filter interface: persistent mean/covariance, predict, and update."""
    def __init__(self, mean, variance, bounds=None):
        self.mean, self.variance, self.bounds = float(mean), float(variance), bounds
        self.check()

    def check(self):
        if not math.isfinite(self.mean) or not math.isfinite(self.variance) or self.variance < 0:
            raise ValueError('Invalid state or scalar covariance')

    def constrain(self):
        before = self.mean
        if self.bounds is not None:
            self.mean = float(np.clip(self.mean, *self.bounds))
            # Retain variance and add squared displacement: conservative heuristic, not truncation.
            self.variance += (self.mean-before)**2
        self.check()
        return self.mean-before

    @abstractmethod
    def predict(self, transition, process_covariance, scale=1.0): pass

    def update(self, observation, measurement_covariance):
        """Apply the shared H=1 Kalman correction in lecture order."""
        if (not math.isfinite(observation) or not math.isfinite(measurement_covariance)
                or measurement_covariance <= 0):
            raise ValueError('Observation and measurement covariance must be finite; covariance positive')
        # Innovation: nu = z - h(mu_bar), with h(x)=x and H=1.
        innovation = observation-self.mean
        # Innovation covariance: S = H Sigma_bar H^T + Q, with scalar H=1.
        innovation_covariance = self.variance+measurement_covariance
        # Kalman gain: K = Sigma_bar H^T S^-1.
        gain = self.variance/innovation_covariance
        # Mean correction: mu = mu_bar + K nu.
        self.mean += gain*innovation
        # Course covariance correction: Sigma = (I-KH) Sigma_bar.
        self.variance = (1-gain)*self.variance
        bound_adjustment = self.constrain()
        return dict(innovation_mm=innovation, S_mm2=innovation_covariance, gain=gain,
                    posterior_bound_adjustment_mm=bound_adjustment)

    def result(self): return self.mean, self.variance

    @staticmethod
    def check_process_covariance(process_covariance):
        if not math.isfinite(process_covariance) or process_covariance < 0:
            raise ValueError('Process covariance must be finite and nonnegative')

class EKF(ScalarFilter):
    def predict(self, transition, process_covariance, scale=1.0, state_jacobian=None):
        self.check_process_covariance(process_covariance)
        # Mean prediction: mu_bar = g(u, mu_previous).
        previous_mean = self.mean
        self.mean = transition(previous_mean)
        # Covariance prediction: Sigma_bar = G Sigma G^T + R.
        jacobian = state_jacobian
        if jacobian is None:
            raise ValueError('EKF requires the analytical state_jacobian')
        self.variance = jacobian*jacobian*self.variance+process_covariance
        adjustment = self.constrain()
        return dict(jacobian=jacobian, sigma_points_clipped=0, prior_bound_adjustment_mm=adjustment)

class UKF(ScalarFilter):
    def predict(self, transition, process_covariance, scale=1.0):
        self.check_process_covariance(process_covariance)
        # n=1, alpha=1, beta=2, kappa=0: lambda=0, positive off-centre weights.
        spread = math.sqrt(self.variance)
        points = np.array([self.mean, self.mean+spread, self.mean-spread])
        clipped = np.clip(points, *self.bounds) if self.bounds is not None else points
        count = int(np.count_nonzero(points != clipped))
        values = np.array([transition(float(x)) for x in clipped])
        self.mean = float((values[1]+values[2])/2)
        delta = values-self.mean
        self.variance = float(2*delta[0]**2 + .5*delta[1]**2 + .5*delta[2]**2 + process_covariance)
        adjustment = self.constrain()
        return dict(jacobian=math.nan, sigma_points_clipped=count, prior_bound_adjustment_mm=adjustment)

class OpenLoop(EKF):
    """Same deterministic mean propagation; runner never calls update."""
