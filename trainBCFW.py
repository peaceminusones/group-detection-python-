import numpy as np 
import random
import math
from constraintFind import constraintFind
from constraint import constraint
from constraint1 import constraint1
import lossGM as loss
import featureMap as fm
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def trainBCFW(X_train, Y_train):
    """
        X_train--------------------------------------------------------------------------------
                   trackid: 当前窗口内（10s内）所有行人的id
                         F: 当前窗口内（10s内）所有帧的所有行人数据（位置、frameid、trackid、速度）
                   couples: 当前窗口内（10s内）所有可能的组（即两两组合）
                myfeatures: 当前窗口内（10s内）所有可能成组的两两轨迹之间形成的特征，包含四个特征【物理距离，相互影响程度，轨迹形状相似性，目标重合度】
        Y_train--------------------------------------------------------------------------------
            从文件中读出的当前窗口内的正确的成组信息
    """
    
    model_w = []
    w_final = []

    if len(X_train) == 0:
        return model_w

    # initialize variables
    n_it = 300  # 迭代次数：500 # maximum number of iterations
    n = len(X_train)
    w = np.zeros((np.array(X_train[0]['myfeatures']).shape[1], 1))
    w_i = np.zeros((np.array(X_train[0]['myfeatures']).shape[1], n))
    l = 0
    l_i = np.zeros(n)

    lambda_c = 1 / 10   # regularization parameter
    w_final = np.zeros((np.array(X_train[0]['myfeatures']).shape[1], 1)) # (2*特征数量,1)大小的零列向量

    for k in range(n_it): # 迭代500次
        # pick a block at random
        decimal = random.random() # 随机产生0-1之间的小数
        iblock = math.ceil(decimal*n) - 1
        # if k > len(i_number)-1:
        #     break
        # iblock = number[k]
        print('---------------------------------------')
        print('k = '+ str(k) +', iblock = ' + str(iblock))
        
        start = time.clock()
        # find the most violated
        model_w = w
        if np.all(model_w == 0):
            iblock = 1
            y_star = constraintFind(model_w, X_train[iblock], Y_train[iblock])
        else:
            y_star = constraint1(model_w, X_train[iblock], Y_train[iblock])
        # y_star = constraint(model_w, X_train[iblock], Y_train[iblock])
        print("y_star: ",y_star)
        
        # find the new best value of the variable
        w_s = (1/lambda_c/n)*(np.array(fm.featureMap(X_train[iblock], Y_train[iblock])) - np.array(fm.featureMap(X_train[iblock], y_star)))
        print("w_s: ",w_s)

        # compute the loss at the new point
        delta,_,_ = loss.lossGM(Y_train[iblock], y_star)
        l_s = (1/lambda_c/n)*delta
        print("l_s: ",l_s)
        
        # compute the step size
        step_size = 2*n/(k+2*n)
        # step_size = ((lambda_c*np.dot(w_i[:,iblock]-w_s, w) + l_s - l_i[iblock] ) / (lambda_c*np.dot(w_i[:,iblock]-w_s, w_i[:,iblock]-w_s)))[0]
        # if math.isnan(step_size):
        #     continue
        # step_size = [step_size, 0][0 > step_size]
        # step_size = [step_size, 1][step_size > 1]
        print("step_size: ",step_size)

        # evaluate w_i and l_i
        w_i_new = (1 - step_size)*w_i[:, iblock] + step_size*w_s
        l_i_new = (1 - step_size)*l_i[iblock] + step_size*l_s

        # update w and l
        w = w + w_i_new.reshape(-1,1) - w_i[:, iblock].reshape(-1,1)  # !列向量相减
        l = l + l_i_new - l_i[iblock]
        print("w: ",w)
        print("l: ",l)

        # update w_i and l_i
        w_i[:, iblock] = w_i_new
        l_i[iblock] = l_i_new

        w = ((k+1)/(k+3)) * model_w + (2/(k+3)) * w
        print('w = \n'+ str(w))

        w_final = np.hstack((w_final, w))

        end = time.clock()
        print("time:",end-start)
        # break
        
    plt.plot(w_final.T)
    plt.show()
    plt.close()
    
    model_w = w
    
    return model_w