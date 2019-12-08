"""
 extract trajectory feature:
"""
import numpy as np
import math
import pandas as pd
from itertools import combinations 
from prox import prox
from granger import granger
from dtw import dtw
from heatmap import heatmap
from v_similar import v_similar
from orientation import orientation
import flatten

def getFeatureFromWindow(myF, index_start, index_end, video_par, model_par):
    # 需要把10s内所有帧的数组提出来
    myF = myF.iloc[index_start:index_end, :]

    # 将10s内所有帧的不重复的track_id提取出来，表示有多少个人
    F = myF.values
    track_id = sorted(set(F[:,1].astype(int)))

    # 计算这10s内每个人在场景中待了多久（多少个frame），如果时间过短（小于4），从当前考虑的数据中删除
    for i in range(len(track_id)):
        if len(myF[myF[1] == track_id[i]]) < 4:
            # 筛选出要删除的数据
            choosebytrackid = myF[myF[1] == track_id[i]]
            # 得到每行的行号，然后逐个删除
            row = choosebytrackid.iloc[:,0].index.values
            myF = myF.drop(index = row)

    # 重新索引，并更新track_id
    myF = myF.reset_index(drop=True)
    # F = myF.values
    track_id = sorted(set(myF.values[:, 1].astype(int)))
    # print(track_id)
    # print(myF)
    """
        PERSONAL FEATURES -----------------------------------------------------------------------------
        (i.e. time, position and velocity)
    """
    """
        path[i] 表示行人i的轨迹信息，是dict类型
        其中数据类型：
                有速度时：{int_path_id: dataframe_path(frameid,position_X,position_Y,v_X,v_Y)}
                无速度时：{int_path_id: dataframe_path(frameid,position_X,position_Y)}
    """
    path = dict()
    # 从10s的第一帧到最后一帧，把每个人的轨迹提取出来，其中数据类型{int_path_id: dataframe_path}
    for t in track_id:
        choosebytrackid = myF[myF[1] == t]
        path[t] = choosebytrackid.iloc[:,[0,2,4]]

    # 如果include_derivatives=true，则提取速度特征
    if model_par.include_derivatives:
        for t in track_id:
            trajectory = path[t].values
            trajectoryPlusVeloc = []
            for i in range(trajectory.shape[0]):
                if i == 0:
                    # 每条轨迹的第一个位置，由于没有之前的位置，所以速度置为nan
                    v_nan = np.insert(trajectory[0], len(trajectory[0]), values=[np.nan, np.nan])
                    trajectoryPlusVeloc.append(list(v_nan))
                else:
                    # 计算之后所有位置的速度(x方向和y方向)
                    vx = trajectory[i, 1] - trajectory[i-1, 1]
                    vy = trajectory[i, 2] - trajectory[i - 1, 2]
                    velocity = np.insert(trajectory[i], len(trajectory[i]), values=[vx, vy])
                    trajectoryPlusVeloc.append(list(velocity))
            # 把含有速度特征的轨迹信息替换到path变量中，其中trajectoryPlusVeloc是array类型，又转换为dataframe类型再替换
            path[t] = pd.DataFrame(np.delete(trajectoryPlusVeloc, 0, 0))  # 删除第一行，因为如果有速度特征的话，第一行的速度值为Nan
    
    # print(path)
    """
        PAIR-WISE FEATURES：-------------------------------------------------------------------------------------

        physical distance |  feature_pd  |  prox
        trajectory shape  |  feature_ts  |  dtw
        motion causality  |  feature_vs  |  velocity
        paths convergence |  feature_pc  |  heatmap
    """
    # 通过track_id先将两两行人形成初始化的组，对于数据集student003大小为：（1128，2）
    couples = group(track_id)
    print(couples.shape[0])
    detectedGroup = group1(path, track_id)
    print(detectedGroup.shape[0])
    
    feature_pd = np.zeros((couples.shape[0], 1))
    feature_ts = np.zeros((couples.shape[0], 1))
    feature_vs = np.zeros((couples.shape[0], 1))
    feature_pc = np.zeros((couples.shape[0], 1))
    
    # compute features for each couple  detectedGroup.shape[0]
    for i in range(couples.shape[0]):
        # 提取出第i行的couple的两个轨迹，数据类型都是dataframe
        traj1 = path[couples[i,0]]
        traj2 = path[couples[i,1]]
        traj_1 = traj1.values
        traj_2 = traj2.values
        
        """
            1) compute proxemics: physical distance | feature_pd 
        """
        if model_par.features[0] == 1:
            traj1_frameid = traj1.iloc[:,0].values
            traj2_frameid = traj2.iloc[:,0].values
            feature_pd[i] = prox(traj1_frameid, traj2_frameid, traj_1, traj_2)
        """
            2) compute MD-DTW: trajectory shape  |  feature_ts
        """
        if model_par.features[1] == 1:
            dist, k= dtw(traj_1, traj_2)
            # 由于距离很大程度上取决于被比较的点的数量，所以它必须被标准化
            feature_ts[i] = dist/k
        """
            3) compute : velocity similarity  |  feature_vs
        """
        if model_par.features[2] == 1:
            # F1 = granger(traj_1, traj_2)
            # F2 = granger(traj_2, traj_1)
            # F1 = granger(traj1, traj2)
            # F2 = granger(traj2, traj1)
            # feature_mc[i] = max(F1, F2)
            # print(feature_mc[i])
            dist, k = v_similar(traj_1, traj_2)
            feature_vs[i] = dist/k
        """
            4) compute HEAT MAPS: paths convergence |  feature_pc
            modification: 利用“匀速直线运动”得到未来运动的方向，然后利用方向夹角得到特征
        """
        # if model_par.features[3] == 1:
        #     allHeatMaps[i], feature_pc[i] = heatmap(traj_1[:, 0:3], traj_2[:, 0:3], video_par)
        #     # if model_par.features[3] != 1:
        # else:
        #     feature_pc[i] = 0
        if model_par.features[3] == 1:
            feature_pc[i] = orientation(traj_1[:, 0:5], traj_2[:, 0:5])
        
        # print(feature_pd[i])
        # print(feature_ts[i])
        # print(feature_vs[i])
        # print(feature_pc[i])
    
    # 把四个特征列向量组合成一个n*4的二维矩阵[feature_pd, feature_ts, feature_mc, feature_pc]
    myfeatures = np.concatenate((feature_pd, feature_ts),axis = 1)
    myfeatures = np.concatenate((myfeatures, feature_vs),axis = 1)
    myfeatures = np.concatenate((myfeatures, feature_pc),axis = 1)

    return [track_id, F, couples, myfeatures, detectedGroup]

def group1(path, track_id):
    couples = np.array(list(combinations(track_id, 2)))
    # dist = []
    friends = []
    for c in couples:
        [frameid0,px0,py0,vx0,vy0] = path[c[0]].iloc[-1].values
        [frameid1,px1,py1,vx1,vy1] = path[c[1]].iloc[-1].values
        d = math.pow(math.pow(px0 - px1 ,2) + math.pow(py0 - py1 ,2), 0.5)
        # dist.append(d)
        if d > 12:  # 换数据集的话需要改！！！！！！！！！！！！！！！！！！！
            continue
        friends.append([c[0],c[1]])
    # a = flatten.flatten(friends)
    # b = track_id
    # singles = list(set(b).difference(set(a)))
    # singles = [[singles[i]] for i in range(len(singles))]
    return np.array(friends)

def group(track_id):
    return np.array(list(combinations(track_id, 2)))