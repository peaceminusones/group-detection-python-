"""
    physical distance |  feature_pd  |  prox
"""
import numpy as np
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prox(traj1_f, traj2_f, traj1, traj2):
    interset_f = sorted(list(set(traj1_f) & set(traj2_f)))
    if len(interset_f) == 0:
        similar = 0
    else:
        inter_traj1 = []
        inter_traj2 = []
        for i in range(len(traj1_f)):
            if traj1_f[i] in interset_f:
                inter_traj1.append(list(traj1[i, 1:3]))
        for i in range(len(traj2_f)):
            if traj2_f[i] in interset_f:
                inter_traj2.append(list(traj2[i, 1:3]))

        inter_t1 = np.array(inter_traj1)
        inter_t2 = np.array(inter_traj2)
        # print("inter_t1:",inter_t1)
        # print("inter_t2:",inter_t2)
        s = square(inter_t1 - inter_t2)
        square_column_sum = s.sum(axis=1)
        square05 = pow_minus(list(square_column_sum), 0.5)
        square05 = -square05
        e = exp(square05)
        ed = e.sum()
        similar = ed / max(len(traj1_f), len(traj2_f))
        # file_name = 'output.txt'
        # with open(file_name,'a') as file_obj:
        #     file_obj.write("inter_t1:\n"+str(inter_t1)+"\n")
        #     file_obj.write("inter_t2:\n"+str(inter_t2)+"\n")
        #     file_obj.write("similar:\n"+str(similar)+"\n")
    return 1 - similar

def square(list_x):
    s = [[list_x[i][j] ** 2 for j in range(len(list_x[i]))] for i in range(len(list_x))]
    return np.array(s)

def pow_minus(list_x, n):
    s = np.array(list(map(lambda x: math.pow(x, n), list_x)))
    return s

def exp(list_x):
    s = np.array(list(map(lambda x: math.exp(x), list_x)))
    return s

# if __name__ == "__main__":

