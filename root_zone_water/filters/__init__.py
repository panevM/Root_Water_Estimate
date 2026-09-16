"""Регистар на филтри што runner.run ги избира според конфигурациското име.

За нов филтер додајте класа во FILTERS и усогласете ги дозволените имиња
во config.load_config/generate_config. runner.compare има сопствен список,
а runner.run посебно го предава аналитичкиот Јакобијан на EKF и OpenLoop.
За математичките чекори и наследената корекција продолжете во scalar.py.
"""
from .scalar import ScalarFilter, EKF, UKF, OpenLoop

FILTERS = {'ekf': EKF, 'ukf': UKF, 'open_loop': OpenLoop}
