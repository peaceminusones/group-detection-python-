
import numpy as np 
from itertools import combinations 
import lossGM as loss
import featureMap as fm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import flatten
import os
import copy
from functools import partial
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parfor(information, i):
    """
    information = [y, couples, n_cluster, X_train, Y_train, model_w]
    """
    y = information[0]
    couples = information[1]
    n_cluster = information[2]
    X_train = information[3]
    Y_train = information[4]
    model_w = information[5]

    couple1 = couples[:, 0]
    couple2 = couples[:, 1]

    Y_temp = list(np.zeros(n_cluster - 1))
    
    # 合并两个特定的cluster
    if len(y[couple1[i]]) > 1 and len(y[couple2[i]]) > 1:
        Y_temp[0] = y[couple1[i]] + y[couple2[i]]
    elif len(y[couple1[i]]) > 1 and len(y[couple2[i]]) == 1:
        Y_temp[0] = y[couple1[i]] + [y[couple2[i]]]
    elif len(y[couple1[i]]) == 1 and len(y[couple2[i]]) > 1:
        Y_temp[0] = [y[couple1[i]]] + y[couple2[i]]
    elif len(y[couple1[i]]) == 1 and len(y[couple2[i]]) == 1:
        Y_temp[0] = [y[couple1[i]], y[couple2[i]]]
    
    k = 0
    for j in range(n_cluster):
        if j != couple1[i] and j != couple2[i]:
            k = k + 1
            Y_temp[k] = y[j]
    
    delta,_,_ = loss.lossGM(Y_train, Y_temp)
    # groud truth
    psi_g = fm.featureMap(X_train, Y_train)
    # train
    psi_t = fm.featureMap(X_train, Y_temp)

    H_temp = (delta + np.dot(model_w.reshape(1,-1), psi_t))[0] 

    return [H_temp, Y_temp]

def constraintFind(model_w, X_train, Y_train):
    # INITIALIZE CLUSTERS FOR GREEDY APPROXIMATION 贪心算法初始化聚类
    """
    参数：
        model_w: 权重，8×1的列向量
        X_train: 某个窗口（i）的训练数据
        Y_train: 某个窗口（i）的训练数据(群组信息)
    """
    # 这个时间窗口下的所有行人id
    # X_train_couples = X_train['couples']
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
            
            # 对与当前窗口下的所有行人组成的cluster，cluster组成的y，计算H(y) = max(Δ(yi, y) - W.TδΨi(y))
            delta,_,_ = loss.lossGM(Y_train, y) # 求Δ(yi, y)
            # print(delta)
            
            # groud truth
            psi_g = fm.featureMap(X_train, Y_train)
            # train
            psi_t = fm.featureMap(X_train, y)
            # print(psi)
            H = (delta + np.dot(model_w.reshape(1,-1), psi_t))[0]   #np.mat(model_w).T*np.mat(psi).T
            
            # 现在这个窗口里所有可能的couple的数量和相应的index组合【即所有cluster两两组合】
            couples = group([i for i in range(n_cluster)])
            # print(couples.shape[0])
            couples = deleteNoCouples(couples, y, X_train['detectedGroups'])
            # print(couples.shape[0])
            # 对于所有可能的clusters，我们必须evaluate H(Y)
            # 用并行做：迭代可以写成与以前的结果无关的形式
            # 每个迭代的集群保存在obj_Y_temp中，每个迭代的结果(分数)保存在H_temp中
            H_temp = np.zeros(couples.shape[0])
            
            obj_Y_temp = dict()

            # 把接下来多线程里需要用到的信息放到一个list里，方便传参
            information = [y, couples, n_cluster, X_train, Y_train, model_w]
            func = partial(parfor,information)
            p = multiprocessing.Pool(8) # 声明了6个线程数量
            iteration = [i for i in range(couples.shape[0])]
            v = p.map(func, iteration)
            p.close()
            p.join()
            
            # func = partial(parfor,information)
            # p = ProcessPoolExecutor(max_workers=10) # 声明了6个进程数量
            # v = p.map(func, range(couples.shape[0]))
            # p.shutdown(wait=True)

            H_temp_and_obj_Y_temp = [r for r in v]
            for i in range(len(H_temp_and_obj_Y_temp)):
                H_temp[i] = H_temp_and_obj_Y_temp[i][0]
                obj_Y_temp[i] = H_temp_and_obj_Y_temp[i][1]

            # for i in range(len(v)):
            #     H_temp[i] = v[i][0]
            #     obj_Y_temp[i] = v[i][1]
            
            # H_max = max(list(H_temp))
            H_max = np.nanmax(H_temp)
            H_max_index = list(H_temp).index(H_max)

            if H_max > H:
                y = obj_Y_temp[H_max_index]
                # loop until something has changed
                changed = True
            # break
    return y

def deleteNoCouples(couples, y, x_train_detectedgroups):
    deleteindex = []
    couple1 = couples[:, 0]
    couple2 = couples[:, 1]
    for i in range(couples.shape[0]):
        c1 = flatten.flatten(y[couple1[i]])
        c2 = flatten.flatten(y[couple2[i]])
        c = [c1, c2]
        mycouples = group(flatten.flatten(c))
        flag = 0
        if len(mycouples) > 0:
            for j in range(len(mycouples)):
                if([mycouples[j][0], mycouples[j][1]] in x_train_detectedgroups) and (flatten.flatten([mycouples[j][0], mycouples[j][1]]) not in c):
                    flag = 1
                    break
        if flag == 0:  # 表示不能合并
            deleteindex.append(i)
    
    couples = np.delete(couples, deleteindex, axis=0)

    return couples

def group(track_id):
    if len(track_id) > 1:
        track_id = flatten.flatten(track_id)
    track_id = sorted(track_id)
    return np.array(list(combinations(track_id, 2)))