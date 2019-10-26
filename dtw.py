"""
    trajectory shape  |  feature_ts  |  dtw

    Dynamic Time Warping Algorithm:
    Dist is unnormalized distance between traj1 and traj2
    D: the accumulated distance matrix
    k: the normalizing factor
    w: the optimal path
    traj1: the vector you are testing against
    traj2: the vector you are testing

    The orinal code by: T. Felty (2005). Dynamic Time Warping [Computer program].
"""

import numpy as np
import pandas as pd
import sys
import numba as nb

@nb.jit
def dtw(traj1, traj2):
    # traj1 & traj2：表示当前couple的两个轨迹，计算两条轨迹之间的dtw
    # eg: traj1(n×m)：n表示轨迹1的位置点的个数，m表示轨迹1的特征个数（5表示包括了速度，3表示不包括速度）
    dist = 0
    
    len_traj1 = traj1.shape[0]
    len_traj2 = traj2.shape[0]
    n_feature = traj1.shape[1]  # traj1的position(x,y)，可能还有速度

    d = 0
    for i in range(n_feature - 1):
        traj1_fi = traj1[:, i + 1] # traj1除了frameid，一列一列提取出来
        traj2_fi = traj2[:, i + 1] # 同上
        # 标准化
        traj1_fi = normalization(traj1_fi)
        traj2_fi = normalization(traj2_fi)

        # print(np.tile(traj1_fi.reshape(-1,1), (1,len_traj2)))
        # print(np.tile(traj2_fi, (len_traj1, 1)))
        d = d + (np.tile(traj1_fi.reshape(-1,1), (1,len_traj2)) - np.tile(traj2_fi, (len_traj1, 1)))**2
    
    d = d**0.5
    D = np.zeros((d.shape[0],d.shape[1]))
    D[0,0] = d[0,0]
    for i in range(1,len_traj1):
        for j in range(len_traj2):
            D1 = D[i-1, j]
            if j > 1:
                D2 = D[i-1, j-1]
            else:
                D2 = float('inf')
            if j > 2:
                D3 = D[i-1, j-2]
            else:
                D3 = float('inf')
            
            D[i, j] = d[i,j] + min([D1, D2, D3])

    dist= D[len_traj1-1, len_traj2-1]
    
    """
        reconstruct the path, so that we can gain information about the 
        minimum number of steps needed to minimize the distance, and thus 
        have a normalization coefficient
    """
    n = len_traj1 - 1
    m = len_traj2 - 1
    k = 1
    while((n+m)!=0):
        if n - 1 == -1:
            m = m - 1
        elif m - 1 == -1:
            n = n - 1
        else:
            min_index = np.argmin([D[n-1, m], D[n, m-1], D[n-1, m-1]])
            if min_index == 0:
                n = n - 1
            elif min_index == 1:
                m = m - 1
            else:
                n = n - 1
                m = m - 1
        k = k + 1
    
    return dist, k

@nb.jit
def normalization(x):
    return (x - np.mean(x))/np.std(x, ddof=1)