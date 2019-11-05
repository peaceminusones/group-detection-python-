"""
    
"""
import numpy as np
from itertools import combinations 
import multiprocessing
from isClusterLegal import isClusterLegal
import featureMap as fm
import lossGM as loss
import flatten

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parfor(information, j):
    """
    information = [i, cluster, X_test, obj_score_temp, obj_cluster_temp, couples, w]
    """
    i = information[0]
    cluster = information[1]
    X_test = information[2]
    obj_score_temp = information[3]
    obj_cluster_temp = information[4]
    couples = information[5]
    w = information[6]
    
    n_cluster = len(cluster)
    # if not isClusterLegal(cluster[couples[j,0]], cluster[couples[j,1]], X_test[str(i)]['detectedGroups']):!!!!!!!!
    if not isClusterLegal(cluster[couples[j,0]], cluster[couples[j,1]], X_test[i]['detectedGroups']):
        return [0, 0]

    cluster_temp = list(np.zeros(n_cluster - 1))
    # if len(cluster[couples[j,0]]) > 1:
    #     cluster[couples[j,0]] = flatten.flatten(cluster[couples[j,0]])
    # if len(cluster[couples[j,1]]) > 1:
    #     cluster[couples[j,1]] = flatten.flatten(cluster[couples[j,1]])
    # cluster_temp[0] = [cluster[couples[j,0]], cluster[couples[j,1]]]
    if len(cluster[couples[j,0]]) > 1 and len(cluster[couples[j,1]]) > 1:
        cluster_temp[0] = cluster[couples[j,0]] + cluster[couples[j,1]]
    elif len(cluster[couples[j,0]]) > 1 and len(cluster[couples[j,1]]) == 1:
        cluster_temp[0] = cluster[couples[j,0]] + [cluster[couples[j,1]]]
    elif len(cluster[couples[j,0]]) == 1 and len(cluster[couples[j,1]]) > 1:
        cluster_temp[0] = [cluster[couples[j,0]]] + cluster[couples[j,1]]
    elif len(cluster[couples[j,0]]) == 1 and len(cluster[couples[j,1]]) == 1:
        cluster_temp[0] = [cluster[couples[j,0]], cluster[couples[j,1]]]

    k = 0
    for m in range(n_cluster):
        if m != couples[j, 0] and m != couples[j, 1]:
            k = k + 1
            cluster_temp[k] = cluster[m]
    
    # psi = fm.featureMap(X_test[str(i)], cluster_temp)!!!!!!!!!!!!!!!!
    psi = fm.featureMap(X_test[i], cluster_temp)
    
    obj_score_temp[j] = np.dot(w.T, psi)[0]
    obj_cluster_temp[j] = cluster_temp
    return [obj_score_temp[j], obj_cluster_temp[j]]

def test_struct_svm(X_test, Y_test, w):
    # this variable will contain our computed clustering
    myY = dict()

    # this variable is going to accumulate the absolute error computed during the test
    error = 0
    p_abs = 0
    r_abs = 0
    perf = np.zeros(len(X_test))
    for i in range(len(X_test)):
        print(i + 1, '/', len(X_test), '.............')
        # start with each element in its own cluster
        # cluster = dict()
        # xi_members = np.array(X_test[str(i)]['trackid'])
        # for j in range(xi_members.shape[0]):
        #     cluster[j] = xi_members[j]
        # % start with each element in its own cluster
        xi_members = X_test[i]['trackid']
        cluster = [xi_members[j] for j in range(len(xi_members))]
        # get the number of clusters
        n_clusters = len(cluster)
        # if there is more than one cluster try to merge two of them!
        changed = True
        while changed and n_clusters > 1:
            changed = False
            # evaluate our current score
            # psi = fm.featureMap(X_test[str(i)], cluster)!!!!!!!!!
            # print(X_test[i])
            # print(Y_test[i])
            # print("------------------------------------")
            psi = fm.featureMap(X_test[i], cluster)
            # print(psi)
            # psi = np.zeros(X_test[i]['myfeatures'].shape[1])
            # # loop through each cluster
            # for m in range(len(cluster)):
            #     # 当前窗口下的cluster再进行两两成组
            #     mycouples = group(cluster[m])
            #     # 对每个cluster计算
            #     if len(mycouples) > 0:
            #         for n in range(len(mycouples)):
            #             index = X_test[i]['couples'].index([mycouples[n,0], mycouples[n,1]])
            #             psi = psi + X_test[i]['myfeatures'][index, :]
            # print(psi)

            obj_score = np.dot(w.T, psi)[0]

            # try all possible joinings...
            couples = group([j for j in range(n_clusters)])
            
            """
                evaluate them all using a parallel for:
                    as a matter of fact the iterations can be written as indipendent from previous results
                The clusters of each iteration will be saved in obj_cluster_temp 
                while result (score) of each iteration will be saved in obj_score_temp
            """
            obj_score_temp = np.zeros(couples.shape[0])
            obj_cluster_temp = dict()

            information = [i, cluster, X_test, obj_score_temp, obj_cluster_temp, couples, w]
            p = multiprocessing.Pool(6) # 声明了6个线程数量
            v = [p.apply_async(parfor, (information, j,)) for j in range(couples.shape[0])]
            p.close()
            p.join()

            score_temp_and_obj_cluster = [r.get() for r in v]
            for j in range(len(score_temp_and_obj_cluster)):
                obj_score_temp[j] = score_temp_and_obj_cluster[j][0]
                obj_cluster_temp[j] = score_temp_and_obj_cluster[j][1]
            
            obj_score_temp_max = max(list(obj_score_temp))
            max_index = list(obj_score_temp).index(obj_score_temp_max)

            if obj_score_temp_max > obj_score:
                cluster = obj_cluster_temp[max_index]
                # print(cluster)
                changed = True

            
            n_clusters = len(cluster)

        myY[i] = cluster
        print(cluster)
        print(Y_test[i])
        # # 处理cluster成lossGM可以处理的格式
        # for j in range(len(cluster)):
        #     if len(cluster[j]) > 1:
        #         cluster[j] = flatten.flatten(cluster[j])
        
        # delta, p, r  = loss.lossGM(cluster, Y_test[str(i)])!!!!!!!!!!
        delta, p, r  = loss.lossGM(cluster, Y_test[i])
        p_abs = p_abs + p
        r_abs = r_abs + r
        error = error + delta
        perf[i] = 2*p*r/(p+r)
        # break
    
    return myY, error, p_abs, r_abs, perf

# def group(track_id):
#     return np.array(list(combinations(track_id, 2)))

def group(track_id):
    if len(track_id) > 1:
        track_id = flatten.flatten(track_id)
    track_id = sorted(track_id)
    return np.array(list(combinations(track_id, 2)))