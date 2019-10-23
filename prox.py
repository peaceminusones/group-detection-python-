

import numpy as np
import pandas as pd
import math

def prox(traj1_f, traj2_f, traj1, traj2):
    interset_f = sorted(list(set(traj1_f) & set(traj2_f)))
    if len(interset_f) == 0:
        similar = 0
    else:
        inter_traj1 = []
        inter_traj2 = []
        for i in range(len(interset_f)):
            t = traj1[traj1[0] == interset_f[i]]
            r = traj2[traj2[0] == interset_f[i]]
            inter_traj1.append(list(t.iloc[:, 1:].values[0]))
            inter_traj2.append(list(r.iloc[:, 1:].values[0]))

        inter_traj1 = np.array(inter_traj1)
        inter_traj2 = np.array(inter_traj2)
        s = square(inter_traj1 - inter_traj2)
        square_column_sum = s.sum(axis=1)
        square05 = pow_minus(list(square_column_sum), 0.5)
        e = exp(square05)
        ed = e.sum()
        similar = ed / max(len(traj1_f), len(traj2_f))
    return similar

def square(list_x):
    s = [[list_x[i][j] ** 2 for j in range(len(list_x[i]))] for i in range(len(list_x))]
    return np.array(s)

def pow_minus(list_x, n):
    return np.array(list(map(lambda x: -math.pow(x, n), list_x)))

def exp(list_x):
    return np.array(list(map(lambda x: math.exp(x), list_x)))