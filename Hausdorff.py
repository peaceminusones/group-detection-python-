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
import sys
import math

def Hausdorff(traj1, traj2):
    # traj1 & traj2：表示当前couple的两个轨迹，计算两条轨迹之间的modified hausdorff距离
    traj1_start = traj1[0,0]
    traj2_strat = traj2[0,0]

    if traj1_start > traj2_strat:
        temp = traj1
        traj1 = traj2
        traj2 = temp
    
    n = traj1.shape[0]  # 轨迹1长度
    m = traj2.shape[0]  # 轨迹2长度

    beta = 0.2
    # a--->b
    distab = 0
    for i in range(n):
        # 对每个轨迹1中的每个点，找到对应轨迹2中距离最短的点
        # 计算轨迹2中每个点到轨迹1第i个点的距离
        dist_ai_to_b = traj2 - traj1[i]
        euclidean = (dist_ai_to_b[:,1]**2 + dist_ai_to_b[:,2]**2)**0.5
        nearest = min(euclidean)
        # 得到轨迹1的第i个点到轨迹2的哪个点距离最近
        sigmai = list(euclidean).index(nearest)
        b_sigmai = traj2[sigmai] # 得到该点的位置信息
        dist_ai_to_bsi = ((traj1[i]-b_sigmai)[1]**2 + (traj1[i]-b_sigmai)[2]**2)**0.5
        vi = traj1[i,3:5]
        vsi = traj2[sigmai,3:5]
        cos = 1 - vi.dot(vsi)/(np.linalg.norm(vi)*np.linalg.norm(vsi))
        if math.isnan(cos):
            cos = 0
        distab = distab + dist_ai_to_bsi + beta * cos
    d_AtoB = distab/n
    
    # b--->a做同样处理
    distba = 0
    for j in range(m):
        # 对每个轨迹1中的每个点，找到对应轨迹2中距离最短的点
        # 计算轨迹2中每个点到轨迹1第i个点的距离
        dist_bj_to_a = traj1 - traj2[j]
        euclidean = (dist_bj_to_a[:,1]**2 + dist_bj_to_a[:,2]**2)**0.5
        nearest = min(euclidean)
        # 得到轨迹1的第i个点到轨迹2的哪个点距离最近
        sigmaj = list(euclidean).index(nearest)
        a_sigmaj = traj1[sigmaj] # 得到该点的位置信息
        dist_bj_to_asj = ((traj2[j]-a_sigmaj)[1]**2 + (traj2[j]-a_sigmaj)[2]**2)**0.5
        vj = traj2[j,3:5]
        vsj = traj1[sigmaj,3:5]
        cos = 1 - vj.dot(vsj)/(np.linalg.norm(vj)*np.linalg.norm(vsj))
        if math.isnan(cos):
            cos = 0
        distba = distba + dist_bj_to_asj + beta * cos
    d_BtoA = distba/m
    fab = min(d_AtoB, d_BtoA)
    
    # FAB = math.exp(-fab)
    # print("FAB:",FAB)

    # # temporal weight --------------------------------------------
    # traj1_end = traj1[-1,0]
    # traj2_end = traj2[-1,0]
    # if traj1_end > traj2_end:
    #     deltad = traj2[-1,0] - traj2[0,0]
    # else:
    #     deltad = traj1[-1,0] - traj2[0,0]
    # yita = min(n,m)/max(n,m)
    # C = deltad*(yita - 0.4)/math.pow(max(n,m),2)
    # # print("deltad:",deltad)
    # # print("yita:",yita)
    # # print("C:", C)
    # w = 1/(1 + math.exp(-C))
    # # w = 1/(1 - C)
    # # print("w:",w)
    # similarity = math.pow(FAB, 1/w)
    # # print("similarity:",similarity)
    # # print(traj1, traj2)
    return fab