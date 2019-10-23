"""
the function is used to access data from different sources.

the output data syntax:
1) informations about the trajectories
   myF = [frame_id, track_id, x, y, z, v_x, v_y, v_z]
2) informations about the clusters
   clusters = cell array of structs.member = [...]
3) informations about the simulation/video
   video_par = video_par.videoObj
               video_par.length
               video_par.downsampling
               video_par.xMin
               video_par.xMax
               video_par.yMin
               video_par.yMax
"""

import os
import pandas as pd          # dataframe
from scipy.io import loadmat # 读取.mat文件
import collections           # flatten
import re                    # 正则

dataDirectory= "mydata/student003"

def loadData(source, snd_parameter):
    # if (source == "generate trajectories"):
    #     [myF, clusters, video_par] = traj(float(snd_parameter))
    # elif (source == "load from file"):
    
    if source == "load from file":
        # search for trajectories.txt, clusters.txt and video_par.mat in dirName and load data from them.
        # for trajectories.txt
        if os.path.exists(snd_parameter + "/trajectories.txt"):
            myF = pd.read_csv(snd_parameter + "/trajectories.txt", sep=' ', header=None)
        else:
            print("Doesn't exist the file: " + snd_parameter + "/trajectories.txt")
            return
        # clusters.txt
        if os.path.exists(snd_parameter + "/clusters.txt"):
            clusters = pd.read_csv(snd_parameter + "/clusters.txt", header=None)
            process_clusters(clusters)
        else:
            print("Doesn't exist the file: " + snd_parameter + "/clusters.txt")
            return
        # video_par.mat
        video_par = dict()
        if os.path.exists(snd_parameter + "/video_par.mat"):
            video_par_dict = loadmat(snd_parameter + "/video_par.mat") # 把video_par.mat数据提取出来，这里video_par_dict是dict类型
            video_par_data = video_par_dict["video_par"]               # 把数据内容提取出来，video_par变成数组
            # 把数据表示的意义提取出来----------------------------------------
            video_par_data_dtype = video_par_data.dtype                
            video_par_keys = re.findall("[a-zA-Z-\_]+", str(video_par_data_dtype))[::2]  # 利用正则表达式(英文字母和下划线)，把数据的key从str中抽取出来
            #----------------------------------------------------------------
            # 把数据的内容与key对应到一起，放到dict中----------------------------
            video_par_data_array = flatten(video_par_data)
            for i in range(len(video_par_data_array)):
                if(str(video_par_data_array[i].dtype)[:4] == "uint"):
                    video_par[video_par_keys[i]] = video_par_data_array[i][0][0]  # 把[[a]]把数字从数组中读出来
                elif(str(video_par_data_array[i].dtype)[:4] == "<U26"):
                    video_par[video_par_keys[i]] = video_par_data_array[i][0]     # 把['a']把字符串从数组中读出来
                else:
                    video_par[video_par_keys[i]] = video_par_data_array[i]        # 3×3的数组，直接赋值
            # -----------------------------------------------------------------
        else:
            print("Doesn't exist the file: " + snd_parameter + "/video_par.mat")
            return
        # picture .jpg
        if os.path.exists(snd_parameter + "/video.avi"):
            video_par['videoObj'] = snd_parameter + "/video.avi"
        else:
            if os.path.exists(snd_parameter + "/000001.jpg"):
                video_par['videoObj'] = snd_parameter + "/%06d.jpg"
            elif os.path.exists(snd_parameter + "/000001.jpeg"):
                video_par['videoObj'] = snd_parameter + "/%06d.jpeg"
            else:
                video_par['videoObj'] = 0
    else:
        print("The input source must be either 'generate trajectories' or 'load from file'")
        return

    return [myF, clusters, video_par]


# 把dataframe clusters里存储的string类型的空格隔开的id，存储到一维int类型的数组中
def process_clusters(clusters):
    for i in range(len(clusters)):
        item = clusters.iloc[i,0]
        clusters.iloc[i,0] = list(map(int,item.split()))
        # print(list(map(int,item.split())))


# 把一个数组套了好几维度的展平
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

# if __name__ == "__main__":
#     myF, clusters, video_par = loadData("load from file", dataDirectory)
#     # print(myF, clusters, video_par)
#     print(myF.iloc[1,2])
    






