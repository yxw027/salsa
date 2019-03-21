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

RawGNSSResType = np.dtype([
	('t', (np.float64, N)),
    ('res', (np.float64, (N, NSAT, 2))),
	('id', (np.int32, (N, NSAT))),
])

def ReadFeatRes(filename):
    f = open(filename)
    featRes = dict()

    while True:
        ti = np.fromfile(f, dtype=np.float64, count=1)

        # EOF
        if len(ti) == 0:
            break

        N = np.fromfile(f, dtype=np.int64, count=1)
        for n in range(N):
            feat_id = np.fromfile(f, dtype=np.int32, count=1)[0]
            n_res = np.fromfile(f, dtype=np.int64, count=1)[0]
            from_idx = np.fromfile(f, dtype=np.int32, count=1)[0]
            if n_res == 0:
                continue
            feat_res = np.fromfile(f, dtype=(np.float64, 4), count=n_res)

            if feat_id not in featRes:
                featRes[feat_id] = {'t' :[], 'res': []}

            featRes[feat_id]['t'].append(ti)
            featRes[feat_id]['res'].append(np.hstack((np.tile(from_idx, (len(feat_res), 1)), feat_res)))
    for key, value in featRes.iteritems():
        featRes[key]['t'] = np.array(value['t'])

    return featRes

def ReadFeat(filename):
    f = open(filename)
    featPos = {'t': []}
    while True:
        t = np.fromfile(f, dtype=np.float64, count=1)
        if len(t) == 0: break # EOF
        N = np.fromfile(f, dtype=np.int64, count=1)
        featPos['t'].append(t)

        for n in range(N):
            feat_id = np.fromfile(f, dtype=np.int32, count=1)[0]
            pos = np.fromfile(f, dtype=(np.float64, 3), count = 1)

            if feat_id not in featPos:
                featPos[feat_id] = []
            featPos[feat_id].append(pos)
    for key, value in featPos.iteritems():
        if key == 't':
            featPos[key] = np.array(value)[:,0]
        else:
            featPos[key] = np.atleast_2d(np.array(value)[:,0,:])
    return featPos





