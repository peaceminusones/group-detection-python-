
import numpy as np 
from itertools import combinations 
import itertools
import lossGM as loss
import featureMap as fm
import multiprocessing
from isClusterLegal import isClusterLegal
import flatten

# from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
# import warnings
# warnings.simplefilter('ignore', category=NumbaWarning)
# from numba import jit
# @jit

def parfor(information, i):
    """
    y = information[0]
    couples = information[1]
    detectedGroups = information[2]
    H_temp = information[3]
    """
    y = information[0]
    couples = information[1]
    detectedGroups = information[2]
    H_temp = information[3]
    obj_Y_temp = information[4]
    n_cluster = information[5]
    X_train = information[6]
    Y_train = information[7]
    model_w = information[8]

    # 把组成couples的两个分开，其中couple1和couple2里存的是他们在y中的index
    # 切片变量c，这样它就可以使用迭代变量来建立索引
    couple1 = couples[:, 0]
    couple2 = couples[:, 1]

    # 先检查两个cluster
    if not isClusterLegal(y[couple1[i]], y[couple2[i]], detectedGroups):
        return [0, 0]
    
    Y_temp = list(np.zeros(n_cluster - 1))
    # 合并两个特定的cluster
    if len(y[couple1[i]]) > 1:
        y[couple1[i]] = flatten.flatten(y[couple1[i]])
    if len(y[couple2[i]]) > 1:
        y[couple2[i]] = flatten.flatten(y[couple2[i]])
    Y_temp[0] = [y[couple1[i]], y[couple2[i]]]

    # cluster counter
    k = 0
    for j in range(n_cluster):
        if j != couple1[i] and j != couple2[i]:
            k = k + 1
            Y_temp[k] = y[j]
    
    delta,_,_ = loss.lossGM(Y_train, Y_temp)
    psi = fm.featureMap(X_train, Y_temp)
    H_temp[i] = (delta + np.mat(model_w).T*np.mat(psi).T)[0, 0]

    obj_Y_temp[i] = Y_temp

    return [H_temp[i], obj_Y_temp[i]]


def constraintFind(model_w, parameter, detectedGroups, X_train, Y_train):
    # INITIALIZE CLUSTERS FOR GREEDY APPROXIMATION 贪心算法初始化聚类
    """
    参数：
        model_w: 权重，8×1的列向量
        parameter: parameter['C'] = 10  # regularization parameter
                   parameter['maxIter'] = 300      # maximum number of iterations
        detectedGroups: X_train里提取出来的，当前窗口内（10s内）所有可能的组（即两两组合）
        X_train: 某个窗口（i）的训练数据
        Y_train: 某个窗口（i）的训练数据(群组信息)
    """
    
    # 这个时间窗口下的所有行人id
    pedestrians = X_train['trackid']
    # 对每个行人都创建一个cluster
    y = [[pedestrians[i]] for i in range(len(pedestrians))]

    # make sure the first iteration verifies the condition
    changed = True

    while changed:
        changed = False
        # print(y)
        n_cluster = len(y) # 当前窗口下cluster的数量，其中y表示“对每个行人都创建一个cluster”

        if n_cluster > 1:  # 如果cluster的数量大于1
            # 现在这个窗口里所有可能的couple的数量和相应的index组合【即所有cluster两两组合】
            couples = group([i for i in range(n_cluster)])
            
            # 对与当前窗口下的所有行人组成的cluster，cluster组成的y，计算H(y) = max(Δ(yi, y) - W.TδΨi(y))
            delta,_,_ = loss.lossGM(Y_train, y) # 求Δ(yi, y)
            # print(delta)
            psi = fm.featureMap(X_train, y)
            # print(psi)
            H = (delta + np.mat(model_w).T*np.mat(psi).T)[0, 0]
            
            # 对于所有可能的clusters，我们必须evaluate H(Y)
            # 用并行做：迭代可以写成与以前的结果无关的形式
            # 每个迭代的集群保存在obj_Y_temp中，每个迭代的结果(分数)保存在H_temp中
            H_temp = np.zeros(couples.shape[0])

            obj_Y_temp = dict()

            # 把接下来多线程里需要用到的信息放到一个list里，方便传参
            information = [y, couples, detectedGroups, H_temp, obj_Y_temp, n_cluster, X_train, Y_train, model_w]
            
            v = []
            p = multiprocessing.Pool(6) # 声明了6个线程数量
            v = [p.apply_async(parfor, (information, i,)) for i in range(couples.shape[0])]
            p.close()
            p.join()
            
            # v = [parfor(information, i) for i in range(couples.shape[0])]

            H_temp_and_obj_Y_temp = [r.get() for r in v]
            for i in range(len(H_temp_and_obj_Y_temp)):
                H_temp[i] = H_temp_and_obj_Y_temp[i][0]
                obj_Y_temp[i] = H_temp_and_obj_Y_temp[i][1]

            # for i in range(len(v)):
            #     H_temp[i] = v[i][0]
            #     obj_Y_temp[i] = v[i][1]

            H_max = max(list(H_temp))
            H_max_index = list(H_temp).index(H_max)
            
            if H_max > H:
                y = obj_Y_temp[H_max_index]
                # loop until something has changed
                changed = True
            
    return y

def group(track_id):
    return np.array(list(combinations(track_id, 2)))

