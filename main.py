import numpy as np
import json 
import random
from loadData import loadData
from loadPreTrained import loadPreTrained
from getFeatureFromWindow import getFeatureFromWindow
from getClustersFromWindow import getClustersFromWindow
from loadfeature import loadfeature
from MyEncoder import MyEncoder
from trainBCFW import trainBCFW
from test import test_struct_svm
from showCluster import showCluster

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataDirectory = "mydata/student003"

class model_par:
    display = False
    window_size = 10
    trainingSetSize = 20
    testingSetSize = 20 - trainingSetSize
    features = [1, 1, 1, 1]
    numberOfFeatures = sum(features)

class model:
    feature_extraction = False
    trainMe = False
    preTrain_w = []
    testMe = False

if __name__ == "__main__":
    """
    initialize, load weight and data -----------------------------------------------------
    """
    # model.preTrain_w = loadPreTrained(model.trainMe, model_par.features, dataDirectory)
    # load data
    [myF, clusters, video_par] = loadData("load from file", dataDirectory)
    print("video_par: \n" ,video_par)
    """
    feature extraction -------------------------------------------------------------------
    """
    index_start = 0
    index_end = 0
    # test_starting_index = 0
    if model.feature_extraction or not(os.path.exists(dataDirectory + "/featureX.json") and os.path.exists(dataDirectory + "/featureY.json")):
        X = dict()   # input data
        Y = dict()   # ground truth info

        print("\nExtracting features: 0%")

        for i in range(model_par.trainingSetSize + model_par.testingSetSize):

            # window_size = 10s, 需要把10s内所有帧的数组提出来，所以需要得到第10s最后一帧的索引
            while (index_end <= myF.shape[0]) and (myF.iloc[index_end, 0] <= (myF.iloc[index_start, 0] + model_par.window_size * video_par['frame_rate'])):
                index_end = index_end + 1
            print("index start: ", index_start," ---> index end: ", index_end)

            # for this window, compute features
            allinformation = getFeatureFromWindow(myF, index_start, index_end, video_par, model_par)
            # print(allinformation)
            X[i] = {'trackid': allinformation[0], 'F': allinformation[1].tolist(), 'couples': allinformation[2], 'myfeatures': allinformation[3].tolist(), 'detectedGroups': allinformation[4]}

            # for this window, retrieve the cluster (so we can use them in the training)
            Y[i] = getClustersFromWindow(X[i]['trackid'], dataDirectory)

            # 移动窗口
            index_start = index_end + 1
            index_end = index_start

            print("Extracting features: " + str(round((i+1) / (model_par.trainingSetSize + model_par.testingSetSize) * 100)) + "%")

        # 把input data和output data保存为json数据格式！
        with open(dataDirectory + '/featureX.json', 'w') as f:
            json.dump(X, f, cls=MyEncoder, indent=4, separators=(',',':'))
        with open(dataDirectory + '/featureY.json', 'w') as f:
            json.dump(Y, f, cls=MyEncoder, indent=4, separators=(',',':'))

    else:
        print("Extraction from featureX")
        # 直接从文件中提取特征
        with open(dataDirectory + '/featureX.json', 'r') as f:
            X1 = json.load(f)
        print("Extraction from featureY1")
        with open(dataDirectory + '/featureY.json', 'r') as f:
            Y = json.load(f)
        X, Y = loadfeature(dataDirectory,"student003.mat")
        print("done!\n")
        for i in range(len(X)):
            X[i]['trackid'] = X1[str(i)]['trackid']
            X[i]['couples'] = X1[str(i)]['couples']
            X[i]['detectedGroups'] = X1[str(i)]['detectedGroups']

        # # ----------------------------------------------------------------------------------------
        # Y = dict()
        # for i in range(model_par.trainingSetSize + model_par.testingSetSize):
        #     while (index_end <= myF.shape[0]) and (myF.iloc[index_end, 0] <= (myF.iloc[index_start, 0] + model_par.window_size * video_par['frame_rate'])):
        #         index_end = index_end + 1

        #     Y[i] = getClustersFromWindow(X[str(i)]['trackid'], dataDirectory)

        # with open(dataDirectory + '/featureY1.json', 'w') as f:
        #     json.dump(Y, f, cls=MyEncoder, indent=4, separators=(',',':'))
        # print("featureY1 -----------------done!\n")
        # # -----------------------------------------------------------------------------------------
        
        # print(np.array(X[0]['myfeatures']))
        # 放弃第三个特征（ganger）
        # if model_par.features == [1, 1, 0, 1]:
        #     for i in range(len(X)):
        #         feature_4 = np.array(X[i]['myfeatures'])
        #         X[i]['myfeatures'] = feature_4[:,[0,1,3]]
        # print(Y[0])
        

        """
        现在已经有了特征 ----------------------------------------------------------------------------
            input data: X 
            output data: Y
        """
        # 特征标准化，找到所有数据四个特征的各自最大值 --------------------------------------
        n_feature = sum(model_par.features)
        mymax = np.zeros((1, n_feature))
        for i in range(model_par.trainingSetSize):
            # arrayX = np.abs(np.array(X[str(i)]['myfeatures']))
            arrayX = np.abs(np.array(X[i]['myfeatures']))
            columnMax = np.nanmax(arrayX, axis=0)  # 得到每一列的最大值
            if columnMax.shape[0] > 0:
                merge = np.vstack((mymax, columnMax))
                mymax = np.nanmax(merge, axis=0)
        # print(mymax)
        # 判断选择的标准化mymax里是否有0，如果有0，则不能作为标准化，需要重新设定
        if 0 in mymax:
            mymax = np.zeros((1, n_feature))
            for i in range(model_par.testingSetSize):
                arrayX = np.abs(np.array(X[i]['myfeatures']))
                columnMax = np.nanmax(arrayX, axis=0)  # 得到每一列的最大值
                if columnMax.shape[0] > 0:
                    merge = np.vstack((mymax, columnMax))
                    mymax = np.nanmax(merge, axis=0)
        # print(mymax)
        # mymax = np.array([0.5776,2.2642,0.0553])
        # # --------------------------------------------------------------------------------
        
        # 特征标准化 & 创建互补特征 --------------------------------------------------------
        for i in range(model_par.trainingSetSize + model_par.testingSetSize):
            # make them similarity measures between 0 and 1
            # length = np.array(X[str(i)]['myfeatures']).shape[0]
            length = np.array(X[i]['myfeatures']).shape[0]
            if  length > 0:
                mymax_array2D = np.array([mymax for _ in range(length)])
                # X[str(i)]['myfeatures'] = 1 - (np.array(X[str(i)]['myfeatures']) / mymax_array2D)
                X[i]['myfeatures'] = (np.array(X[i]['myfeatures']) / mymax_array2D)
                X[i]['myfeatures'][:,1] = 1 - X[i]['myfeatures'][:,1]

            # 创建互补的特征，以更好地识别相似阈值
            # for j in range(model_par.numberOfFeatures):
            #     # value = X[str(i)]['myfeatures'][:, j] - 1
            #     # X[str(i)]['myfeatures'] = np.insert(X[str(i)]['myfeatures'], model_par.numberOfFeatures + j, values=value, axis=1)
            #     value = X[i]['myfeatures'][:, j] - 1
            #     X[i]['myfeatures'] = np.insert(X[i]['myfeatures'], model_par.numberOfFeatures + j, values=value, axis=1)
        
        print(X[0]['couples'].index([9,273]))
        index = X[0]['couples'].index([9,273])
        print(X[0]['myfeatures'][index])
        # print(np.array(X[0]['detectedGroups']).shape)
        print("data: " + dataDirectory + ", training: " + str(model_par.trainingSetSize) + ", testing: " + str(model_par.testingSetSize))
        # # --------------------------------------------------------------------------------
        
        """
        # Training -----------------------------------------------------------------------
        # """
        allSetSize = model_par.testingSetSize + model_par.trainingSetSize
        # testindex = random.sample(range(0,allSetSize),model_par.testingSetSize)
        allindex = [x for x in range(allSetSize)]
        # trainindex = list(set(allindex).difference(set(testindex)))
        # print(testindex, trainindex)
        # X_test = [X[str(i)] for i in allindex]
        # Y_test = [Y[str(i)] for i in allindex]
        # X_train = [X[str(i)] for i in allindex]
        # Y_train = [Y[str(i)] for i in allindex]
        X_test = [X[i] for i in allindex]
        Y_test = [Y[i] for i in allindex]
        X_train = [X[i] for i in allindex]
        Y_train = [Y[i] for i in allindex]

        if model.trainMe:
            print("\nTraining the classifier on the training set:\n")
            modelBCFW_weight= trainBCFW(X_train, Y_train)
            print("modelBCFW_weight=\n", modelBCFW_weight)
        else:
            # modelBCFW_weight = model.preTrain_w
            # modelBCFW_weight = np.array([[-0.12815193],[ 0.02901785],[ 0.02662733],[ 0.05865231],[-0.1421993 ],[ 0.01497048],[ 0.01257996],[ 0.04460494]])
            # modelBCFW_weight = np.array([[-0.11746883],[ 0.07694488],[ 0.01152641],[-0.04012392],[-0.15165551],[ 0.0427582 ],[-0.02266027],[-0.07431061]])
            # modelBCFW_weight = np.array([[0.24849833],[0.09128355],[-0.07110776],[0.11287341]])
            modelBCFW_weight = np.array([[ 0.04886871],[ 0.06247873],[ 0.06336363],[ 0.06000938],[-0.08224337],[-0.06863335],[-0.06774844],[-0.0711027 ]])
            print("\nLoad model weight: " + str(modelBCFW_weight.T[0]))

        """
        Testing ----------------------------------------------------------
        """
        if model.testMe:
            print("\nTesting the classifier on the testing set:\n")
            # print(Y_test)
            myY_test, absolute_error_test, p_test, r_test, perf = test_struct_svm(X_test, Y_test, modelBCFW_weight)
            print("myY_test: \n", myY_test)
            print(absolute_error_test)
            print(p_test)
            print(r_test)
            print(perf)
            # output_test = np.round(np.array([absolute_error_test, p_test, r_test])/model_par.testingSetSize*100*100)/100
            output_test = np.round(np.array([absolute_error_test, p_test, r_test])/allSetSize*100*100)/100
            print('Testing error: ',output_test[0], '%% (Precision: ',output_test[1],'%%, Recall: ',output_test[2],'%%)\n')
        else:
            print("\nLoading myY_test:")
            

        if model_par.display:
            showCluster(myF, Y_test, myY_test, model_par, video_par, dataDirectory)

