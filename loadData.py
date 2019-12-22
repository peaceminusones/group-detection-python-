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
import numpy as np 

dataDirectory= "mydata/GVEII/GVEII"

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
    

class DataLoader():
    def __init__(self, datasets, col):
        self.dataframe = dict()
        self.col = col
        for i in range(len(datasets)):
            # print()# loadData("load from file", datasets[i])
            if os.path.exists(datasets[i] + "/trajectories.txt"):
                self.dataframe[i] = pd.read_csv(datasets[i] + "/trajectories.txt", sep=' ', header=None)
            else:
                print("Doesn't exist the file: " + datasets[i] + "/trajectories.txt")
                return
        # print(self.dataframe)
    
    def get_train_data(self,seq_len):
        x = []
        y = []
        for i in range(len(self.dataframe)):
            '''
                循环把每个数据集中的轨迹信息都提取出来
            '''
            df = self.dataframe[i]
            # 把当前数据集的的行人id都提取出来
            track_id = sorted(set(df.values[:,1].astype(int)))
            # 计算每个人在场景中待了多久（多少个frame），如果时间过短
            # 就不够lstm训练，则从数据中删除
            for j in range(len(track_id)):
                if len(df[df[1] == track_id[j]]) < 14:
                    # 筛选出要删除的数据
                    choosebytrackid = df[df[1] == track_id[j]]
                    # 得到每行的行号，然后逐个删除
                    row = choosebytrackid.iloc[:,0].index.values
                    df = df.drop(index = row)
            # 重新索引，并更新track_id
            df = df.reset_index(drop=True)
            track_id = sorted(set(df.values[:, 1].astype(int)))

            # 根据track_id提取该数据集下的所有轨迹
            for t in track_id:
                choosebytrackid = df[df[1] == t]
                path = choosebytrackid.iloc[:,[2,4]] # 提取position(x,y)
                traj = path.values
                # 提取速度信息
                trajectoryPlusVeloc = []
                for i in range(traj.shape[0]):
                    if i == 0:
                        # 每条轨迹的第一个位置，由于没有之前的位置，所以速度置为nan
                        v_nan = np.insert(traj[0], len(traj[0]), values=[np.nan, np.nan])
                        trajectoryPlusVeloc.append(list(v_nan))
                    else:
                        # 计算之后所有位置的速度(x方向和y方向)
                        vx = traj[i, 0] - traj[i-1, 0]
                        vy = traj[i, 1] - traj[i-1, 1]
                        velocity = np.insert(traj[i], len(traj[i]), values=[vx, vy])
                        trajectoryPlusVeloc.append(list(velocity))
                # 把含有速度特征的轨迹信息替换到path变量中，其中trajectoryPlusVeloc是array类型，又转换为dataframe类型再替换
                path = pd.DataFrame(np.delete(trajectoryPlusVeloc, 0, 0))  # 删除第一行，因为如果有速度特征的话，第一行的速度值为Nan
                
                # for p in range(len(path)-seq_len):
                #     x.append()
            break


        return x,y





