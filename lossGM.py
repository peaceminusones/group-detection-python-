"""
    G-MITRE loss Δ(yi, y) computation
"""

import numpy as np 
import itertools
import flatten
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
# import warnings
# warnings.simplefilter('ignore', category=NumbaWarning)

def lossGM(Y_train, ybar):
    """
    参数：
        ybar: 当前窗口下所有行人的各自的id
        Y_train: 当前窗口下已知的所有组
    """
    # 把当前窗口下已知的所有组y的行人id提取出来
    # group_pedestrian_id在y中的位置给了它一个唯一的标识符
    # Y_train是list类型，里面的元素，有的长度是1，有的长度是2、3、4……，所以group_pedestrian_id就是把Y_train里的元素展开成一个一维向量
    # group_pedestrian_id = list(itertools.chain.from_iterable(Y_train))
    group_pedestrian_id = flatten.flatten(Y_train)
    # UF data structure -------------------------------------------------------------------------------
    # 最终作用是：相应位置上如果是同一个group，就赋予相同的值
    # eg:             Y_train = [[7,8],[15,16,24],...]
    #     group_pedestrian_id = [7,8,15,16,24,...]
    #               UF_ytrain = [1,1,2,2,2,...]
    # UF_ybar同理
    # 最终得到的connected_ytrain是把UF_ytrain重复的和零元素删除；connected_ybar同理
    UF_ytrain = np.zeros(2*len(group_pedestrian_id))
    UF_ybar = np.zeros(2*len(group_pedestrian_id))

    for i in range(len(Y_train)):
        for j in range(len(Y_train[i])):
            # find the index of the group_pedestrian_id and update its input in UF_ytrain
            # print(Y_train[i][j])
            # print(group_pedestrian_id)
            if isinstance(Y_train[i][j],list):
                if Y_train[i][j][0] in group_pedestrian_id:
                    res = group_pedestrian_id.index(Y_train[i][j][0])
                    UF_ytrain[res] = i + 1 # 同一组的标号相同
            else:
                if Y_train[i][j] in group_pedestrian_id:
                    res = group_pedestrian_id.index(Y_train[i][j])
                    UF_ytrain[res] = i + 1 # 同一组的标号相同
    
    for i in range(len(ybar)):
        for j in range(len(ybar[i])):
            if isinstance(ybar[i][j],list):
                if ybar[i][j][0] in group_pedestrian_id:
                    res = group_pedestrian_id.index(ybar[i][j][0])
                    UF_ybar[res] = i + 1
            else:
                if ybar[i][j] in group_pedestrian_id:
                    res = group_pedestrian_id.index(ybar[i][j])
                    UF_ybar[res] = i + 1
    connected_ytrain = np.unique(np.array(UF_ytrain))[1:]
    connected_ybar = np.unique(np.array(UF_ybar))[1:]
    # --------------------------------------------------------------------------------------------------
    
    # if we have singletons, we have to add a relationship with "themselfes"
    for i in range(len(connected_ytrain)):
        if sum(UF_ytrain == connected_ytrain[i]) == 1:
            UF_ytrain[len(group_pedestrian_id) + list(UF_ytrain).index(connected_ytrain[i])] = connected_ytrain[i]

    for i in range(len(connected_ybar)):
        if sum(UF_ybar == connected_ybar[i]) == 1:
            UF_ybar[len(group_pedestrian_id) + list(UF_ybar).index(connected_ybar[i])] = connected_ybar[i]


    # now we can apply the MITRE measure, first from y to ybar ...
    num = 0
    den = 0
    for i in range(len(connected_ytrain)):
        index = enumerateIndex(connected_ytrain[i], UF_ytrain)
        num = num + len(index) - len(np.unique([UF_ybar[i] for i in index]))
        den = den + len(index) - 1
    if den == 0:
        R_ytrain_ybar = 1
    else:
        R_ytrain_ybar = num/den

    # apply the MITRE measure, from ybar to y...
    num = 0
    den = 0
    for i in range(len(connected_ybar)):
        index = enumerateIndex(connected_ybar[i], UF_ybar)
        num = num + len(index) - len(np.unique([UF_ytrain[i] for i in index]))
        den = den + len(index) - 1
    if den == 0:
        R_ybar_ytrain = 1
    else:
        R_ybar_ytrain = num/den

    # compute delta ---------------------------------------------------------------------------
    if np.isnan(R_ytrain_ybar) or np.isnan(R_ybar_ytrain):
        F = 0
    elif R_ytrain_ybar + R_ybar_ytrain == 0:
        F = 0
    else:
        F = 2 * R_ytrain_ybar * R_ybar_ytrain /(R_ytrain_ybar + R_ybar_ytrain)

    delta = 1 - F

    # R_ytrain_ybar, R_ybar_ytrain, 
    return delta, R_ybar_ytrain, R_ytrain_ybar

def enumerateIndex(target, a):
    b = []
    for index, nums in enumerate(a):
        if nums == target:
            b.append(index)
    return b