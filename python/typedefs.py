import numpy as np

N = 10

OptType = np.dtype([('t', (np.float64, N)),
                    ('x', (np.float64, (N, 7))),
                    ('v', (np.float64, (N, 3))),
                    ('tau', (np.float64, (N, 2))),
                    ('imu_bias', (np.float64, 6)),
                    ('dt_mocap', np.float64),
                     ])

StateType = np.dtype([('t', np.float64),
                      ('x', (np.float64, 7)),
                      ('v', (np.float64, 3))])