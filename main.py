import os
import numpy as np
import json 
from loadData import loadData
from loadPreTrained import loadPreTrained
from getFeatureFromWindow import getFeatureFromWindow
from getClustersFromWindow import getClustersFromWindow
from loadfeature import loadfeature
from MyEncoder import MyEncoder
from trainBCFW import trainBCFW
from test_struct_svm import test_struct_svm
from showCluster import showCluster

dataDirectory = "mydata/student003"
feature_extraction = True

class model_par:
    display = True
    window_size = 10
    trainingSetSize = 10
    testingSetSize = 20 - trainingSetSize
    features = [1, 1, 1, 1]  # [PROX, GRNG, MDWT, HMAP]

    numberOfFeatures = sum(features)
    include_derivatives = True
    useHMGroupDetection = False

class model:
    trainMe = False
    preTrain_w = []


if __name__ == "__main__":
    """
    initialize, load weight and data -----------------------------------------------------
    """
    model.preTrain_w = loadPreTrained(model.trainMe, model_par.features, dataDirectory)
    # load data
    [myF, clusters, video_par] = loadData("load from file", dataDirectory)
    print("video_par: \n" ,video_par)
    """
    feature extraction -------------------------------------------------------------------
    """
    index_start = 0
    index_end = 0
    # test_starting_index = 0
    if feature_extraction: #or not(os.path.exists(dataDirectory + "/featureX.json") and os.path.exists(dataDirectory + "/featureY.json")):
        X = dict()   # input data
        Y = dict()   # ground truth info

        print("\nExtracting features: 0%\n")

        for i in range(model_par.trainingSetSize + model_par.testingSetSize):

            # if i == model_par.trainingSetSize + 10:
            #     test_starting_index = index_start

            # window_size = 10s, 需要把10s内所有帧的数组提出来，所以需要得到第10s最后一帧的索引
            while (index_end <= myF.shape[0]) and (myF.iloc[index_end, 0] <= (myF.iloc[index_start, 0] + model_par.window_size * video_par['frame_rate'])):
                index_end = index_end + 1
            print("index start --> end: ",index_start, index_end)
            # for this window, compute features
            allinformation = getFeatureFromWindow(myF, index_start, index_end, video_par, model_par)
            X[i] = {'trackid': allinformation[0], 'F': allinformation[1].tolist(), 'couples': allinformation[2].tolist(), 'myfeatures': allinformation[3].tolist(), 'detectedGroups': allinformation[4]}

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
            X = json.load(f)
        print("Extraction from featureY")
        with open(dataDirectory + '/featureY.json', 'r') as f:
            Y = json.load(f)
        print("done!\n")

        print(X[str(1)]['myfeatures'])
        print(Y)
        """
        现在已经有了特征 --------------------------------------------------------------
            input data: X 
            output data: Y
        """
        # 特征标准化，找到所有数据四个特征的各自最大值 --------------------------------------
        mymax = np.zeros((1, 4))
        for i in range(model_par.trainingSetSize):
            arrayX = np.abs(np.array(X[str(i)]['myfeatures']))
            columnMax = np.amax(arrayX, axis=0)  # 得到每一列的最大值
            if columnMax.shape[0] > 0:
                merge = np.vstack((mymax, columnMax))
                mymax = np.amax(merge, axis=0)
        # print(mymax)
        # 判断选择的标准化mymax里是否有0，如果有0，则不能作为标准化，需要重新设定
        if 0 in mymax:
            mymax = np.zeros((1, 4))
            for i in range(model_par.testingSetSize):
                arrayX = np.abs(np.array(X[str(i)]['myfeatures']))
                columnMax = np.amax(arrayX, axis=0)  # 得到每一列的最大值
                if columnMax.shape[0] > 0:
                    merge = np.vstack((mymax, columnMax))
                    mymax = np.amax(merge, axis=0)
        # print(mymax)
        # --------------------------------------------------------------------------------

        # 特征标准化 & 创建互补特征
        for i in range(model_par.trainingSetSize + model_par.testingSetSize):
            # make them similarity measures between 0 and 1
            length = np.array(X[str(i)]['myfeatures']).shape[0]
            if  length > 0:
                mymax_array2D = np.array([mymax for _ in range(length)])
                X[str(i)]['myfeatures'] = 1 - (np.array(X[str(i)]['myfeatures']) / mymax_array2D)

            # 创建互补的特征，以更好地识别相似阈值
            for j in range(model_par.numberOfFeatures):
                value = X[str(i)]['myfeatures'][:, j] - 1
                X[str(i)]['myfeatures'] = np.insert(X[str(i)]['myfeatures'], model_par.numberOfFeatures, values=value, axis=1)
        # print(list(X[str(1)]['myfeatures']))
        print("data: " + dataDirectory + ", training: " + str(model_par.trainingSetSize) + ", testing: " + str(model_par.testingSetSize))
        """
        Training ----------------------------------------------------------
        """
        X_test = X
        Y_test = Y
        X_train = [X[str(i)] for i in range(model_par.trainingSetSize)]
        Y_train = [Y[str(i)] for i in range(model_par.trainingSetSize)]

        if model.trainMe:
            print("\nTraining the classifier on the training set:\n")
            modelBCFW_weight= trainBCFW(X_train, Y_train)
        else:
            # modelBCFW_weight = model.preTrain_w
            # modelBCFW_weight = np.array([[-0.00482523],[ 0.00467267],[ 0.005967  ],[-0.0016145 ],[-0.0047506 ],[ 0.0028309 ],[ 0.00153657],[-0.00796133]])
            modelBCFW_weight = np.array([[-0.0356655111340553],[0.0246113342594141],[0.00524166773229231],[-0.0124222568704902],[-0.0442202270483166],[0.0160566183451527],[-0.00331304818196912],[-0.0209769727847515]])
            print("\nLoad model weight: " + str(modelBCFW_weight.T[0]))

        """
        Testing ----------------------------------------------------------
        """

        print("\nTesting the classifier on the testing set:\n")
        # model_par.testingSetSize = model_par.testingSetSize + model_par.trainingSetSize
        # myY_test, absolute_error_test, p_test, r_test, perf = test_struct_svm (X_test, Y_test, modelBCFW_weight)
        # print(myY_test)
        # print(absolute_error_test)
        # print(p_test)
        # print(r_test)
        # print(perf)

        # output_test = np.round(np.array([absolute_error_test, p_test, r_test])/model_par.testingSetSize*100*100)/100

        myY_test = {0: [[18, 19, 20], [28, 30, 27, 29, 6, 2], [73, 74], [10, 12, 11, 13], [215, 216], [219, 220, 217], [9, 273], [77, 78], [7, 8], [14, 16], [213, 214], [1], [3], [4], [5], [15], [17], [21], [22], [23], [24], [25], [26], [31], [75], [76], [218], [274], [409]],
                    1: [[24, 420, 419, 37], [217, 220], [4, 75], [77, 78], [35, 36], [33, 34], [14, 16], [27, 29], [19, 20], [28, 30], [12, 13, 10], [3], [5], [7], [8], [9], [11], [17], [23], [25], [26], [31], [32], [73], [74], [76], [79], [80], [81], [82], [83], [219], [315], [410], [411]],
                    2: [[233, 410], [38, 39], [80, 81], [300, 301], [217, 220], [84, 85, 82], [40, 302], [303, 304], [28, 30], [11, 12, 13, 41, 10, 45, 44, 76], [27, 29, 19, 20, 232, 87], [4, 75], [33], [34], [35], [36], [37], [42], [43], [74], [77], [78], [79], [83], [86], [219], [221], [222], [223], [305], [315], [316], [317], [411]],
                    3: [[48, 49], [91, 410], [232, 233], [223, 224, 226], [4, 75], [316, 317], [38, 39], [300, 301], [306, 307, 312], [229, 230], [45, 46, 47], [308, 309], [303, 304], [318, 319], [94, 95, 220, 219, 217, 93, 92], [84, 85, 82], [37], [40], [41], [42], [43], [44], [50], [51], [80], [83], [86], [87], [88], [89], [90], [221], [222], [225], [227], [228], [234], [235], [275], [302], [340], [341], [411]],
                    4: [[38, 39, 301], [88, 316, 234, 235], [310, 312], [92, 93], [343, 376, 344], [57, 233, 232], [94, 95], [308, 309], [89, 90], [229, 230], [48, 49], [223, 226], [55, 56, 54], [4, 75], [45, 46, 47], [222, 340], [42], [44], [50], [51], [52], [53], [83], [87], [91], [96], [221], [224], [225], [227], [228], [231], [300], [311], [313], [317], [318], [319], [341], [342]],
                    5: [[51, 412, 228], [46, 47, 45, 102], [89, 90, 342], [94, 95, 229], [87, 344], [52, 53], [60, 61], [222, 340, 343], [55, 56], [232, 233], [98, 99], [50], [54], [57], [58], [59], [72], [92], [93], [96], [97], [100], [101], [225], [230], [231], [311], [313], [341], [376]],
                    6: [[60, 61], [232, 233], [236, 237], [348, 349], [422, 423, 421], [64, 65], [62, 63], [345, 346], [58, 59], [413, 414, 412], [98, 99], [55, 56], [222, 340, 343], [424, 425, 426], [68, 69], [51], [57], [66], [67], [72], [96], [100], [101], [102], [103], [104], [231], [347], [376]],
                    7: [[108, 109], [350, 351], [240, 241], [68, 69], [64, 66, 65], [222, 343], [106, 107], [232, 233], [421, 422, 423], [70, 71], [62, 63, 424], [345, 346], [98, 99], [243, 413], [72, 103], [51], [67], [104], [105], [110], [111], [112], [113], [236], [237], [238], [239], [242], [276], [314], [320], [340], [347], [352], [353], [376], [425], [426]],
                    8: [[244, 245, 243], [236, 237], [64, 66], [114, 116, 115], [123, 124, 122], [72, 103], [232, 233, 238], [351, 353, 350, 352], [378, 379], [120, 121], [246, 247, 422, 423, 67, 421, 426, 425], [428, 429], [345, 346], [65], [70], [71], [104], [106], [107], [108], [109], [110], [111], [112], [113], [117], [118], [119], [242], [320], [321], [322], [323], [376], [377], [427]],
                    9: [[246, 247], [130, 133], [248, 249], [382, 383], [243, 323], [251, 252], [384, 385, 386], [125, 126], [244, 245], [115, 116, 114, 117], [137, 138, 139], [121, 123], [378, 379], [232, 233, 238], [118, 119], [72], [103], [110], [120], [122], [124], [127], [128], [129], [131], [132], [134], [135], [136], [250], [321], [322], [324], [325], [376], [377], [380], [381]],
                    10:[[248, 249], [140, 141], [253, 254], [137, 138], [244, 245], [324, 325], [130, 133, 132], [232, 233, 238], [257, 258, 115, 116, 256], [246, 247, 125, 126, 384], [382, 383], [143, 144], [121, 123], [122], [124], [127], [128], [129], [131], [134], [135], [136], [139], [142], [243], [250], [251], [252], [255], [277], [321], [322], [323], [326], [327], [376], [386]],
                    11:[[144, 277], [148, 149, 249], [121, 123, 138, 137], [127, 134], [324, 325], [130, 132, 133], [253, 254], [356, 357], [128], [129], [140], [141], [142], [143], [145], [146], [147], [232], [233], [238], [248], [256], [257], [258], [259], [326], [327], [328], [354], [355], [376], [386], [387]],
                    12:[[146, 147, 256], [148, 149], [391, 392], [324, 325], [356, 357], [144, 261, 277], [232, 354, 238, 355], [331, 332, 330], [156, 157, 248, 155], [326, 327], [142], [143], [145], [150], [151], [152], [153], [154], [233], [249], [254], [259], [260], [328], [329], [376], [387], [388], [389], [390], [393]],
                    13:[[150, 151], [354, 355], [393, 395], [415, 416], [358, 359], [232, 238, 233], [248, 249], [391, 392], [163, 164, 162], [329, 332, 331], [142], [152], [153], [154], [155], [156], [157], [158], [159], [160], [161], [256], [259], [260], [261], [330], [334], [376], [394], [396], [397]],
                    14:[[160, 358], [156, 158], [280, 281, 279, 282], [248, 249], [165, 166], [169, 170], [360, 361], [331, 332, 329], [142], [155], [159], [161], [162], [163], [164], [167], [168], [256], [259], [260], [262], [278], [333], [334], [335], [336], [359], [393], [396], [397], [398], [399]],
                    15:[[160, 399], [171, 172], [360, 361], [156, 158], [335, 336], [279, 280, 281, 278], [329, 331, 332], [358, 398], [176, 177], [362, 363], [142], [155], [165], [166], [167], [168], [169], [170], [173], [174], [175], [256], [282], [283], [284], [285], [286], [333], [334], [364], [430], [431], [432]],
                    16:[[142, 432], [176, 177], [185, 333], [362, 363], [186, 282, 184, 183, 336], [168], [169], [171], [172], [173], [174], [175], [178], [179], [180], [181], [182], [286], [287], [288], [334], [335], [360], [361], [364]],
                    17:[[183, 184], [142, 432], [191, 192], [189, 190], [362, 363], [176, 177], [169], [178], [179], [180], [181], [182], [185], [186], [187], [188], [194], [287], [288], [364], [365], [366], [400], [417], [418]],
                    18:[[367, 368], [191, 192], [194, 195], [183, 184, 289, 290, 199], [362, 363], [179], [187], [189], [190], [193], [196], [197], [198], [200], [201], [202], [263], [291], [365], [366], [400], [401], [402], [433]],
                    19:[[191, 192], [189, 190, 212, 433], [401, 402], [208, 209, 267, 265, 264, 266, 366], [210, 211], [193], [198], [200], [201], [202], [203], [204], [205], [206], [207], [263], [291], [337], [362], [363], [369], [370], [371]]}

        if model_par.display:
            showCluster(myF, Y_test, myY_test, model_par, video_par)

        


