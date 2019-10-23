
import numpy as np
import pandas as pd

def dtw(traj1, traj2):
    dist = 0
    k = 1
    len_traj1 = len(traj1)
    len_traj2 = len(traj2)
    d = 0
    for i in range(2):
        traj1_position = traj1.iloc[:, i + 1].values
        traj2_position = traj2.iloc[:, i + 1].values

    return [dist, k]