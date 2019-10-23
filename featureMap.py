"""

"""
import numpy as np
from itertools import combinations 
import itertools
import collections
import flatten

# from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
# import warnings
# warnings.simplefilter('ignore', category=NumbaWarning)
# from numba import jit
# @jit

def featureMap(X_train, y):
    """
    参数：
        y: 当前窗口下所有行人的各自的id
        X_train: 当前窗口下所有轨迹的特征
    """
    psi = np.zeros(X_train['myfeatures'].shape[1])
    # loop through each cluster
    for i in range(len(y)):
        # 当前窗口下的cluster再进行两两成组
        mycouples = group(y[i])
        # 对每个cluster计算
        if len(mycouples) > 0:
            for j in range(len(mycouples)):
                index = X_train['couples'].index([mycouples[j,0], mycouples[j,1]])
                psi = psi + X_train['myfeatures'][index, :]
    
    
    return psi

def group(track_id):
    if len(track_id) > 1:
        track_id = flatten.flatten(track_id)
    track_id = sorted(track_id)
    return np.array(list(combinations(track_id, 2)))
