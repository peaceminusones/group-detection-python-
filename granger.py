"""
    compute trajectory feature: motion causality
"""
import numpy as np
import pandas as pd
import math
import scipy.signal as signal
from fractions import Fraction
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

def granger(traj1, traj2, g):
    traj1_f = traj1.iloc[:, 0].values
    traj2_f = traj2.iloc[:, 0].values
    interset_f = sorted(list(set(traj1_f) & set(traj2_f)))
    if len(interset_f) == 0:
        F = 0
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
        fn = inter_traj1.shape[1]
        length = inter_traj1.shape[0]

        mean_traj1 = np.mean(inter_traj1, axis=0)
        mean_traj2 = np.mean(inter_traj2, axis=0)

        inter_traj1 = inter_traj1 - np.array([mean_traj1 for _ in range(length)])
        inter_traj2 = inter_traj2 - np.array([mean_traj2 for _ in range(length)])

        max_traj1 = np.max(abs(inter_traj1), axis=0)
        max_traj2 = np.max(abs(inter_traj2), axis=0)
        max_traj1_2D = np.array([max_traj1 for _ in range(length)])
        max_traj2_2D = np.array([max_traj2 for _ in range(length)])
        inter_traj1 = inter_traj1 / max_traj1_2D
        inter_traj2 = inter_traj2 / max_traj2_2D

        for i in range(fn):
            inter_traj1[:, i] = signal.medfilt(inter_traj1[:, i], 7)
            inter_traj2[:, i] = signal.medfilt(inter_traj2[:, i], 7)

        if g > length - 1:
            g = length - 1
        m = length - g
        X_p = np.zeros((fn, m))
        for i in range(m):
            X_p[:, i] = inter_traj1[i + g, :]

        X = np.zeros((fn * g, m))
        for i in range(m):
            temp = inter_traj1[i:i + g, :].reshape((1, fn * g))
            X[:, i] = temp[0]

        A = np.mat(X_p) * np.linalg.pinv(X)
        errorA = X_p - np.mat(A) * np.mat(X)

        XY = np.zeros((fn * g * 2, m))
        for i in range(m):
            temp = (np.insert(inter_traj1[i:i + g, :], inter_traj1[i:i + g, :].shape[0], inter_traj2[i:i + g, :], axis=0)).reshape((1, fn * g * 2))
            XY[:, i] = temp

        B = np.mat(X_p) * np.linalg.pinv(XY)
        errorB = X_p - np.mat(B) * np.mat(XY)

        S_x = np.zeros((fn, fn))
        S_xy = np.zeros((fn, fn))
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(m):
            S_x = S_x + errorA[:, i] * errorA[:, i].T / (m - 1)
            S_xy = S_xy + errorB[:, i] * errorB[:, i].T / (m - 1)

        a = np.linalg.det(S_x)
        b = np.linalg.det(S_xy)
        if a == 0.0 and b == 0.0:
            return 0.0
        if a == 0.0 and b != 0.0:
            return 100.0
        if a != 0.0 and b == 0.0:
            return 100.0
        if np.isnan(a) or np.isnan(b):
            return 0.0
        c = abs(float(Fraction(a / b)))
        d = math.log(c)
        F = abs(d)

    return F