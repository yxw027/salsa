import numpy as np

N = 10
NSAT = 20

OptType = np.dtype([
	('t', (np.float64, N)),
    ('x', (np.float64, (N, 7))),
    ('v', (np.float64, (N, 3))),
    ('tau', (np.float64, (N, 2))),
    ('imu_bias', (np.float64, 6)),
    ('s', (np.float64, 20)),
])

StateType = np.dtype([
	('t', np.float64),
    ('x', (np.float64, 7)),
    ('v', (np.float64, 3)),
    ('b', (np.float64, 6)),
    ('tau', (np.float64, 2))
])

RawGNSSResType = np.dtype([
	('t', (np.float64, N)),
    ('res', (np.float64, (N, NSAT, 2))),
	('id', (np.int32, (N, NSAT))),
])
