import numpy as np 
import random
import math
import constraintFind as cf
import lossGM as loss
import featureMap as fm
import matplotlib.pyplot as plt
import time

def trainBCFW(X_train, Y_train):
    """
        X_train--------------------------------------------------------------------------------
                   trackid: 当前窗口内（10s内）所有行人的id
                         F: 当前窗口内（10s内）所有帧的所有行人数据（位置、frameid、trackid、速度）
                   couples: 当前窗口内（10s内）所有可能的组（即两两组合）
                myfeatures: 当前窗口内（10s内）所有可能成组的两两轨迹之间形成的特征，包含四个特征【物理距离，相互影响程度，轨迹形状相似性，目标重合度】
            detectedGroups: 即couples
        Y_train--------------------------------------------------------------------------------
            从文件中读出的当前窗口内的正确的成组信息
    """
    
    model_w = []
    w_final = []

    if len(X_train) == 0:
        return model_w
    
    # pattern = dict()
    # labels = dict()

    # for i in range(len(X_train)):
    #     pattern[i] = X_train[i]
    #     labels[i] = Y_train[i]

    # initial parameter
    parameter = dict()
    parameter['C'] = 10             # regularization parameter
    parameter['maxIter'] = 400      # maximum number of iterations

    detectedGroups = [X_train[i]['detectedGroups'] for i in range(len(X_train))]
    # F = [X_train[i]['F'] for i in range(len(X_train))]

    n = len(X_train)
    n_it = parameter['maxIter']

    # initialize variables
    w = np.zeros((np.array(X_train[0]['myfeatures']).shape[1], 1))
    w_i = np.zeros((np.array(X_train[0]['myfeatures']).shape[1], n))

    l = 0
    l_i = np.zeros(n)

    # 开启一个画图的窗口
    # plt.ion()                  
    # plt.figure(1)

    lambda_c = 1 / parameter['C']
    w_final = np.zeros((np.array(X_train[0]['myfeatures']).shape[1], 1)) # (8,1)的零列向量

    for k in range(n_it):
        # pick a block at random
        decimal = random.random() # 随机产生0-1之间的小数
        iblock = math.ceil(decimal*n) - 1
        print('\n---------------------------------------')
        print('k = '+ str(k) +', iblock = ' + str(iblock))

        # find the most violated
        model_w = w
        y_star = cf.constraintFind(model_w, parameter, detectedGroups[iblock], X_train[iblock], Y_train[iblock])
        print("y_star: ")
        print(y_star)
        
        # compute the loss at the new point
        # l_s = ((1/lambda_c)/n)*loss.lossGM(Y_train[iblock], y_star)
        delta,_,_ = loss.lossGM(Y_train[iblock], y_star)
        l_s = (1/n)*delta
        # print(l_s)

        # find the new best value of the variable
        w_s = ((1/lambda_c)/n)*(np.array(fm.featureMap(X_train[iblock], Y_train[iblock])) - np.array(fm.featureMap(X_train[iblock], y_star)))
        # print(w_s)

        # compute the step size
        step_size = ((lambda_c * np.dot(w_i[:,iblock]-w_s, w) + l_s - l_i[iblock] ) / lambda_c / np.dot(w_i[:,iblock]-w_s, w_i[:,iblock]-w_s))[0]
        if math.isnan(step_size):
            continue
        step_size = [step_size, 0][0 > step_size]
        step_size = [step_size, 1][step_size > 1]
        print("step_size: ")
        print(step_size)

        # evaluate w_i and l_i
        l_i_new = (1 - step_size)*l_i[iblock] + step_size*l_s
        w_i_new = (1 - step_size)*w_i[:, iblock] + step_size*w_s
        # print(l_i_new)
        # print(w_i_new)

        # update w and l
        l = l + l_i_new - l_i[iblock]
        w = w + w_i_new.reshape(-1,1) - w_i[:, iblock].reshape(-1,1)  # !列向量相减
        # print(l)
        # print(w)

        # update w_i and l_i
        l_i[iblock] = l_i_new
        w_i[:, iblock] = w_i_new
        # print(l_i)
        # print(w_i)

        w = (k/(k+2)) * model_w + (2/(k+2)) * w
        print('w = \n'+ str(w))

        w_final = np.hstack((w_final, w))
        # print('w_final =')
        # print(w_final)
        
        # plot figure
        # plt.plot(w_final.T)        # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.draw()
        # time.sleep(1)

        # break
        
    plt.plot(w_final.T)
    plt.show()
    plt.close()
    
    model_w = w
    # print(model_w)
    
    return model_w