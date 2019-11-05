
"""

"""

import os
import pandas as pd          # dataframe
from scipy.io import loadmat # 读取.mat文件
import collections           # flatten
import re                    # 正则

dataDirectory= "mydata/student003"

def loadfeature(dataDirectory, filename):
    # if (source == "generate trajectories"):
    #     [myF, clusters, video_par] = traj(float(snd_parameter))
    # elif (source == "load from file"):
    X = dict()
    Y = dict()
    detectedgroup = dict()
    if os.path.exists(dataDirectory + "/" + filename):
        feature_dict = loadmat(dataDirectory + "/" + filename)
        X_all = feature_dict["X"][0]
        for i in range(len(X_all)):
            detectedgroup[0] = X_all[i][4][0][0].tolist()
            X[i] = {'trackid': X_all[i][0].tolist(), 'F': X_all[i][1].tolist(), 'couples': X_all[i][2].tolist(), 'myfeatures': X_all[i][3], 'detectedGroups': detectedgroup} 
        print("-----------------")
        
        Y_all = feature_dict["Y"][0]
        for i in range(len(Y_all)):
            Y_i = [(Y_all[i][0][0][j].tolist())[0] for j in range(len(Y_all[i][0][0]))]
            for j in range(len(Y_i)):
                if len(Y_i[j])>1:
                    Y_i[j] = [[Y_i[j][m]] for m in range(len(Y_i[j]))]
            Y[i] = Y_i
    else:
        print("The file ", dataDirectory + "/" + filename, "doesn't exist")
    
    return X, Y


    
    # if source == "load from file":
    #     # search for trajectories.txt, clusters.txt and video_par.mat in dirName and load data from them.
    #     # for trajectories.txt
    #     if os.path.exists(snd_parameter + "/trajectories.txt"):
    #         myF = pd.read_csv(snd_parameter + "/trajectories.txt", sep=' ', header=None)
    #     else:
    #         print("Doesn't exist the file: " + snd_parameter + "/trajectories.txt")
    #         return
    #     # clusters.txt
    #     if os.path.exists(snd_parameter + "/clusters.txt"):
    #         clusters = pd.read_csv(snd_parameter + "/clusters.txt", header=None)
    #         process_clusters(clusters)
    #     else:
    #         print("Doesn't exist the file: " + snd_parameter + "/clusters.txt")
    #         return
    #     # video_par.mat
    #     video_par = dict()
    #     if os.path.exists(snd_parameter + "/video_par.mat"):
    #         video_par_dict = loadmat(snd_parameter + "/video_par.mat") # 把video_par.mat数据提取出来，这里video_par_dict是dict类型
    #         video_par_data = video_par_dict["video_par"]               # 把数据内容提取出来，video_par变成数组
    #         # 把数据表示的意义提取出来----------------------------------------
    #         video_par_data_dtype = video_par_data.dtype                
    #         video_par_keys = re.findall("[a-zA-Z-\_]+", str(video_par_data_dtype))[::2]  # 利用正则表达式(英文字母和下划线)，把数据的key从str中抽取出来
    #         #----------------------------------------------------------------
    #         # 把数据的内容与key对应到一起，放到dict中----------------------------
    #         video_par_data_array = flatten(video_par_data)
    #         for i in range(len(video_par_data_array)):
    #             if(str(video_par_data_array[i].dtype)[:4] == "uint"):
    #                 video_par[video_par_keys[i]] = video_par_data_array[i][0][0]  # 把[[a]]把数字从数组中读出来
    #             elif(str(video_par_data_array[i].dtype)[:4] == "<U26"):
    #                 video_par[video_par_keys[i]] = video_par_data_array[i][0]     # 把['a']把字符串从数组中读出来
    #             else:
    #                 video_par[video_par_keys[i]] = video_par_data_array[i]        # 3×3的数组，直接赋值
    #         # -----------------------------------------------------------------
    #     else:
    #         print("Doesn't exist the file: " + snd_parameter + "/video_par.mat")
    #         return
    #     # picture .jpg
    #     if os.path.exists(snd_parameter + "/video.avi"):
    #         video_par['videoObj'] = snd_parameter + "/video.avi"
    #     else:
    #         if os.path.exists(snd_parameter + "/000001.jpg"):
    #             video_par['videoObj'] = snd_parameter + "/%06d.jpg"
    #         elif os.path.exists(snd_parameter + "/000001.jpeg"):
    #             video_par['videoObj'] = snd_parameter + "/%06d.jpeg"
    #         else:
    #             video_par['videoObj'] = 0
    # else:
    #     print("The input source must be either 'generate trajectories' or 'load from file'")
    #     return

    # return 


# # 把dataframe clusters里存储的string类型的空格隔开的id，存储到一维int类型的数组中
# def process_clusters(clusters):
#     for i in range(len(clusters)):
#         item = clusters.iloc[i,0]
#         clusters.iloc[i,0] = list(map(int,item.split()))
#         # print(list(map(int,item.split())))


# # 把一个数组套了好几维度的展平
# def flatten(x):
#     result = []
#     for el in x:
#         if isinstance(x, collections.Iterable) and not isinstance(el, str):
#             result.extend(flatten(el))
#         else:
#             result.append(el)
#     return result

# if __name__ == "__main__":
#     loadfeature(dataDirectory, "student003.mat")
