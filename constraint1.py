import numpy as np 
from itertools import combinations 
import lossGM as loss
import featureMap as fm
import flatten
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def constraint1(model_w, X_train, Y_train):
    # INITIALIZE CLUSTERS FOR GREEDY APPROXIMATION 贪心算法初始化聚类
    """
    参数：
        model_w: 权重，8×1的列向量
        X_train: 某个窗口（i）的训练数据
        Y_train: 某个窗口（i）的训练数据(群组信息)
    """
    # 维护一个关系匹配度网络
    couples_features = X_train['myfeatures']
    # 并更新所有行人之间的关系值（n×8）×（8×1），值越大表示越匹配
    score_network = np.dot(np.array(couples_features), model_w) 
    
    # 这个时间窗口下的所有行人id
    pedestrians = X_train['trackid']
    # 对每个行人都创建一个cluster，即当前每个人都是独立的行人
    y = [[pedestrians[i]] for i in range(len(pedestrians))]

    n_cluster = len(y)
    changed = True
    while changed and n_cluster > 1:
        n_cluster = len(y)
        # 对与当前窗口下的所有行人组成的cluster，cluster组成的y，计算H(y) = max(Δ(yi, y) + W.T*Ψi(y))
        delta,_,_ = loss.lossGM(Y_train, y) # 求Δ(yi, y)
        # psi_g = fm.featureMap(X_train, y)  # 求φ(xi, y_train)
        psi = fm.featureMap(X_train, y)  # 求φ(xi, y)
        H = (delta + np.dot(model_w.reshape(1,-1), psi))[0]  # 求H(y)
        
        # 得到关系匹配网络中，当前值最大的couples: max_couples
        max_score = np.nanmax(score_network)
        max_index = list(score_network).index(max_score)
        max_couples = X_train['couples'][max_index]
        
        # 将max_couples合并
        if ([max_couples[0]] in y) and ([max_couples[1]] in y):
            index0 = y.index([max_couples[0]])
            index1 = y.index([max_couples[1]])
        elif ([max_couples[0]] not in y) and ([max_couples[1]] in y):
            for i in range(len(y)):
                if ([max_couples[0]] in y[i]):
                    index0 = y.index(y[i])
                    break
            index1 = y.index([max_couples[1]])
        elif ([max_couples[0]] in y) and ([max_couples[1]] not in y):
            index0 = y.index([max_couples[0]])
            for i in range(len(y)):
                if ([max_couples[1]] in y[i]):
                    index1 = y.index(y[i])
                    break
        else:
            for i in range(len(y)):
                if ([max_couples[0]] in y[i]):
                    index0 = y.index(y[i])
                if ([max_couples[1]] in y[i]):
                    index1 = y.index(y[i])
                
        # 合并两个特定的cluster
        if index0 != index1:
            Y_temp = list(np.zeros(n_cluster - 1))
            if len(y[index0]) > 1 and len(y[index1]) > 1:
                Y_temp[0] = y[index0] + y[index1]
            elif len(y[index0]) > 1 and len(y[index1]) == 1:
                Y_temp[0] = y[index0] + [y[index1]]
            elif len(y[index0]) == 1 and len(y[index1]) > 1:
                Y_temp[0] = [y[index0]] + y[index1]
            elif len(y[index0]) == 1 and len(y[index1]) == 1:
                Y_temp[0] = [y[index0], y[index1]]
            k = 0
            for j in range(n_cluster):
                if j != index0 and j != index1:
                    k = k + 1
                    Y_temp[k] = y[j]
            
        # 对与当前窗口下的所有行人组成的cluster，cluster组成的y，计算H(y) = W.T*φi(y)
        delta,_,_ = loss.lossGM(Y_train, Y_temp) # 求Δ(yi, y)
        # psi_g = fm.featureMap(X_train, y)  # 求φ(xi, y_train)
        psi = fm.featureMap(X_train, Y_temp)  # 求φ(xi, y)
        H_new = (delta + np.dot(model_w.reshape(1,-1), psi))[0]  # 求H(y)
        if H_new >= H:
            y = Y_temp
            changed = True
        else:
            changed = False
        score_network[max_index] = np.nan #因为当前couple已合并
    
    return y
