from .scalar import ScalarFilter, EKF, UKF, OpenLoop

FILTERS = {'ekf': EKF, 'ukf': UKF, 'open_loop': OpenLoop}
