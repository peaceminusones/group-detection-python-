import numpy as np
import math
import pandas as pd
from itertools import combinations 
from prox import prox
from granger import granger
from dtw import dtw
from heatmap import heatmap
# from detectGroups import detectGroups
from v_similar import v_similar

def getTrajFromWindow(myF, index_start, index_end, video_par, model_par):
    # 需要把10s内所有帧的数组提出来
    myF = myF.iloc[index_start:index_end, :]

    # 将10s内所有帧的不重复的track_id提取出来，表示有多少个人
    F = myF.values
    track_id = sorted(set(F[:,1].astype(int)))
    
    # 计算这10s内每个人在场景中待了多久（多少个frame），如果时间过短（小于4），从当前考虑的数据中删除
    for i in range(len(track_id)):
        if len(myF[myF[1] == track_id[i]]) < 5:
            # 筛选出要删除的数据
            choosebytrackid = myF[myF[1] == track_id[i]]
            # 得到每行的行号，然后逐个删除
            row = choosebytrackid.iloc[:,0].index.values
            myF = myF.drop(index = row)
    
    # 重新索引，并更新track_id
    myF = myF.reset_index(drop=True)
    # F = myF.values
    track_id = sorted(set(myF.values[:, 1].astype(int)))
    
    '''
        extract velocity of trajectory of each pedestrain
    '''
    path = dict()
    # 从10s的第一帧到最后一帧，把每个人的轨迹提取出来，其中数据类型{int_path_id: dataframe_path}
    for t in track_id:
        choosebytrackid = myF[myF[1] == t]
        path[t] = choosebytrackid.iloc[:,[0,2,4]]

    # 提取速度特征
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
    
    return path,track_id