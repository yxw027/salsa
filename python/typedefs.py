import numpy as np
import yaml

params = yaml.load(open("../params/salsa.yaml"))

CurrentStateType = np.dtype([
	('t', np.float64),
    ('x', (np.float64, 7)),
    ('v', (np.float64, 3)),
    ('b', (np.float64, 6)),
    ('tau', (np.float64, 2))
])

StateType = np.dtype([
    ('t', np.float64),
    ('x', (np.float64, 7)),
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
    ('BUF_SIZE', np.int32),
    ('head', np.int32),
    ('tail', np.int32),
    ('x', (OptStateType, params["state_buf_size"])),
    # ('s', (np.float64, 0)),
    ('imu', (np.float64, 6))
])


FtType = np.dtype([
    ('id', np.int32),
    ('p', (np.float64, 3)),
    ('rho', np.float64)
])

nf = 10
FeatType = np.dtype([
    ('t', np.float64),
    ('size', np.uint64),
    ('ft', (FtType, nf))
])

ResType2 = np.dtype([
    ('to_node', np.int32),
    ('t', np.float64),
    ('res', (np.float64, 2))
])
nw = 3
ResType1 = np.dtype([
    ('id', np.int32),
    ('size', np.int32),
    ('from_node', np.int32),
    ('to', (ResType2, params["N"]))
])
FeatResType = np.dtype([
    ('t', np.float64),
    ('size', np.int32),
    ('f', (ResType1, params["num_feat"]))
])




