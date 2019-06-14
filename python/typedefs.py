import numpy as np
import yaml

params = yaml.load(open("../params/salsa.yaml"))

XType = np.dtype([
    ('p', (np.float64, 3)),
    ('q', (np.float64, 4))
])

CurrentStateType = np.dtype([
	('t', np.float64),
    ('xu', XType),
    ('vu', (np.float64, 3)),
    ('b', (np.float64, 6)),
    ('tau', (np.float64, 2)),
    ('x', XType),
    ('euler', (np.float64, 3)),
    ('v', (np.float64, 3)),
])

SimStateType = np.dtype([
	('t', np.float64),
    ('x', XType),
    ('euler', (np.float64, 3)),
    ('v', (np.float64, 3)),
    ('b', (np.float64, 6)),
    ('tau', (np.float64, 2)),
    ('x_e2n', XType),
    ('x_b2c', XType),
    ('multipath', np.int32),
    ('denied', np.int32),
    ('mp', (np.float64, params["num_sat"]))
])

StateType = np.dtype([
    ('t', np.float64),
    ('x', XType),
    ('v', (np.float64, 3)),
    ('tau', (np.float64, 2)),
    ('kf', np.int32),
    ('node', np.int32)
])


OptStateType = np.dtype([
    ('node', np.int32),
    ('kf', np.int32),
    ('t', np.float64),
    ('p', (np.float64, 3)),
    ('q', (np.float64, 4)),
    ('v', (np.float64, 3)),
    ('tau', (np.float64, 2))
])

OptType = np.dtype([
    ('head', np.int32),
    ('tail', np.int32),
    ('x', (OptStateType, params["max_node_window"])),
    # ('s', (np.float64, 0)),
    ('imu', (np.float64, 6)),
    ('x_b2c', XType),
    ('x_e2n', XType),
])

SwParamsType = np.dtype([
    ('p', (np.dtype([
        ('t', np.float64),
        ('s', np.float64)]), params["num_sat"]), params["max_node_window"])
])


FtType = np.dtype([
    ('id', np.int32),
    ('p', (np.float64, 3)),
    ('rho', np.float64),
    ('rho_true', np.float64),
    ('slide_count', np.int32)
])

FeatType = np.dtype([
    ('t', np.float64),
    ('size', np.uint64),
    ('ft', (FtType, params["num_feat"]))
])

ResType2 = np.dtype([
    ('to_node', np.int32),
    ('t', np.float64),
    ('res', (np.float64, 2))
])
ResType1 = np.dtype([
    ('id', np.int32),
    ('size', np.int32),
    ('from_node', np.int32),
    ('to', (ResType2, params["max_node_window"]))
])
FeatResType = np.dtype([
    ('t', np.float64),
    ('size', np.int32),
    ('f', (ResType1, params["num_feat"]))
])

MocapResType = np.dtype([
    ('t', np.float64),
    ('size', np.int32),
    ('r', (np.dtype([
        ('t', np.float64),
        ('res', (np.float64, 6))
    ]), params["max_node_window"]))
])

GnssResType = np.dtype([
    ('t', np.float64),
    ('size', np.int32),
    ('r', (np.dtype([
        ('size', np.int32),
        ('v', np.dtype([
            ('t', np.float64),
            ('res', (np.float64, 2))
        ]), params["num_sat"])
    ]), params["max_node_window"]))
])

SatPosType = np.dtype([
    ('t', np.float64),
    ('size', np.int32),
    ('sats', (np.dtype([
        ('id', np.int32),
        ('p', (np.float64, 3)),
        ('v', (np.float64, 3)),
        ('tau', (np.float64, 2)),
        ('azel', (np.float64, 2))
    ]), params["num_sat"]))
])

PRangeResType = np.dtype([
    ('t', np.float64),
    ('size', np.int32),
    ('sats', (np.dtype([
        ('id', np.int32),
        ('res', (np.float64, 3)),
        ('z', (np.float64, 3)),
        ('zhat', (np.float64, 3))
    ]), params["num_sat"]))
])

ImuType = np.dtype([
    ('t', np.float64),
    ('acc', (np.float64, 3)),
    ('omega', (np.float64, 3))
])







