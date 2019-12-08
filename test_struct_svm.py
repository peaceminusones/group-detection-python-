"""
    
"""
import numpy as np
from itertools import combinations 
import multiprocessing
import featureMap as fm
import lossGM as loss
import flatten

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parfor(information, j):
    """
    information = [i, cluster, X_test, couples, w]
    """
    i = information[0]
    cluster = information[1]
    X_test = information[2]
    couples = information[3]
    w = information[4]
    
    n_cluster = len(cluster)

    cluster_temp = list(np.zeros(n_cluster - 1))
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
    
    psi = fm.featureMap(X_test[i], cluster_temp)
    
    obj_score_temp = np.dot(w.T, psi)[0]
    
    return [obj_score_temp, cluster_temp]

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
        xi_members = X_test[i]['trackid']
        cluster = [[xi_members[j]] for j in range(len(xi_members))]
        # get the number of clusters
        n_clusters = len(cluster)
        # if there is more than one cluster try to merge two of them!
        changed = True
        while changed and n_clusters > 1:
            changed = False

            print(cluster)
            psi = fm.featureMap(X_test[i], cluster)

            obj_score = np.dot(w.T, psi)[0]
            # try all possible joinings...
            couples = group([j for j in range(n_clusters)])
            # couples = deleteNoCouples(couples, cluster, X_test[i]['detectedGroups'])
            """
                evaluate them all using a parallel for:
                    as a matter of fact the iterations can be written as indipendent from previous results
                The clusters of each iteration will be saved in obj_cluster_temp 
                while result (score) of each iteration will be saved in obj_score_temp
            """
            obj_score_temp = np.zeros(couples.shape[0])
            obj_cluster_temp = dict()

            func = partial(parfor,information)
            p = multiprocessing.Pool(6) # 声明了6个线程数量
            iteration = [i for i in range(couples.shape[0])]
            v = p.map(func, iteration)
            p.close()
            p.join()
            
            # information = [i, cluster, X_test, couples, w]
            # p = multiprocessing.Pool(6) # 声明了6个线程数量
            # v = [p.apply_async(parfor, (information, j,)) for j in range(couples.shape[0])]
            # p.close()
            # p.join()

            score_temp_and_obj_cluster = [r for r in v]
            for j in range(len(score_temp_and_obj_cluster)):
                obj_score_temp[j] = score_temp_and_obj_cluster[j][0]
                obj_cluster_temp[j] = score_temp_and_obj_cluster[j][1]
            
            obj_score_temp_max = max(list(obj_score_temp))
            max_index = list(obj_score_temp).index(obj_score_temp_max)

            if obj_score_temp_max > obj_score:
                cluster = obj_cluster_temp[max_index]
                changed = True

            n_clusters = len(cluster)

        myY[i] = cluster
        print("----------------------------------")
        print(cluster)
        print(Y_test[i])
        print("----------------------------------")

        delta, p, r  = loss.lossGM(cluster, Y_test[i])
        p_abs = p_abs + p
        r_abs = r_abs + r
        error = error + delta
        perf[i] = 2*p*r/(p+r)
        # break
    
    return myY, error, p_abs, r_abs, perf

def deleteNoCouples(couples, y, x_test_detectedgroups):
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
                if([mycouples[j][0], mycouples[j][1]] in x_test_detectedgroups) and (flatten.flatten([mycouples[j][0], mycouples[j][1]]) not in c):
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