
# import json
# import numpy as np

# data = dict()
# a = [1e-5,0.024583569845257874,-4,5,7,8,3,43,65,45,0.0,14,67]
# b = np.zeros((4,4))
# c = np.ones((5,3))
# d = np.array([[5,7,1],[7,3,2],[9,7,6],[1,0,13]])

# e = [0,35,62,13,7,8,3,43,65,45,0,14,67]
# f = np.ones((4,4))
# g = np.zeros((5,3))
# h = np.array([[0.34234,1.454363,2.654534],[6.34532,0.345345,0.00032],[4.53458,56.7345,5.64536],[1.4534,42.45346,1.67533]])

# for i in range(0,2,2):
#     data[i] = {'trackid': a, 'F': b.tolist(), 'couples': c.tolist(), 'myfeatures': d.tolist()}
#     data[i+1] = {'trackid': e, 'F': f.tolist(), 'couples': g.tolist(), 'myfeatures': h.tolist()}

# with open('data.json','w') as f:
#     json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))



# import numpy as np 
# import collections
# import itertools
# from itertools import combinations 


# def group(track_id):
#     if len(track_id) > 1:
#         track_id = flatten(track_id)
#     return np.array(list(combinations(track_id, 2)))

# def flatten(x):
#     result = list(itertools.chain.from_iterable(x))
#     # for el in x:
#     #     if isinstance(x, collections.Iterable) and not isinstance(el, str):
#     #         result.extend(flatten(el))
#     #     else:
#     #         result.append(el)
#     return result


# a = group([])
# print(a)

# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from math import *
 
# # plt.ion() #开启interactive mode 成功的关键函数
# # plt.figure(1)
# t = np.linspace(0, 20, 100)
 
# for i in range(20):
# 	# plt.clf() # 清空画布上的所有内容。此处不能调用此函数，不然之前画出的轨迹，将会被清空。
#     y = np.sin(t*i/10.0)
#     plt.plot(t, y) # 一条轨迹
#     plt.show()
#     plt.close()
#     # plt.draw()#注意此函数需要调用
#     # plt.pause(1)
#     # time.sleep(1)
# import multiprocessing

# def a(x):
#     return x**2


# if __name__ == "__main__":
    
#     for k in range(10):
#         v = []
#         p = multiprocessing.Pool(6) # 声明了6个线程数量
#         v = [p.apply_async(a, (i+k,)) for i in range(8)]
#         p.close()
#         p.join()
#         print([r.get() for r in v])
    
#     # H_temp_and_obj_Y_temp = [r.get() for r in v]
#     # print(H_temp_and_obj_Y_temp)


# import matplotlib.pyplot as plt

# ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
# ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
# plt.ion()                  # 开启一个画图的窗口
# for i in range(100):       # 遍历0-99的值
# 	ax.append(i)           # 添加 i 到 x 轴的数据中
# 	ay.append(i**2)        # 添加 i 的平方到 y 轴的数据中
# 	plt.clf()              # 清除之前画的图
# 	plt.plot(ax,ay)        # 画出当前 ax 列表和 ay 列表中的值的图形
# 	plt.pause(0.1)         # 暂停一秒
# 	plt.ioff()             # 关闭画图的窗口

# import numpy as np
# from scipy.linalg import solve
# a = np.array([[1, 1, 7], [2, 3, 5], [4, 2, 6]])
# b = np.array([2, 3, 4])
# x1 = solve(a, b)
# # x2 = np.mat(b) * np.mat(a).I
# x2 = b/a
# print(x1,x2)
# import flatten

# print(flatten.flatten([[1],[2]]))

# import numpy as np
# from itertools import combinations 
# import itertools
# import collections
# import flatten

# def group(track_id):
#     if len(track_id) > 1:
#         track_id = flatten.flatten(track_id)
#     track_id = sorted(track_id)
#     return np.array(list(combinations(track_id, 2)))


# print(group([[1],[2],[3]]))
# print(group([1,2,3]))

# import numpy as np
# import cupy as cp
# import time
 
# x=np.zeros((1024,512,4,4))*1024.
# y=np.zeros((1024,512,4,1))*512.3254
# time1=time.time()
# for i in range(20):
#     z=x*y
# print('average time for 20 times cpu:',(time.time()-time1)/20.)
 
# x=cp.zeros((1024,512,4,4))*1024.
# y=cp.zeros((1024,512,4,1))*512.3254
# z=cp.zeros((10,20))
# a= cp.max(cp.array([[2,4,5,6],[4,0,6,7]]),axis=0)
# print(a)
# time1=time.time()
# for i in range(20):
#     z=x*y
# print('average time for 20 times gpu:',(time.time()-time1)/20.)

# import tensorflow as tf
# import timeit

# with tf.device('/cpu:0'):
#    cpu_a = tf.random.normal([10000, 1000])
#    cpu_b = tf.random.normal([1000, 2000])
#    print(cpu_a.device, cpu_b.device)

# with tf.device('/gpu:0'):
#    gpu_a = tf.random.normal([10000, 1000])
#    gpu_b = tf.random.normal([1000, 2000])
#    print(gpu_a.device, gpu_b.device)

# def cpu_run():
#    with tf.device('/cpu:0'):
#       c = tf.matmul(cpu_a, cpu_b)
#    return c

# def gpu_run():
#    with tf.device('/gpu:0'):
#       c = tf.matmul(gpu_a, gpu_b)
#    return c

# # warm up 
# cpu_time = timeit.timeit(cpu_run, number=10)
# gpu_time = timeit.timeit(gpu_run, number=10)
# print('warmup:', cpu_time, gpu_time)

# cpu_time = timeit.timeit(cpu_run, number=10)
# gpu_time = timeit.timeit(gpu_run, number=10)
# print('run time:', cpu_time, gpu_time)


import matplotlib.pyplot as plt
import numpy as np
import time
from math import *

# plt.ion() #开启interactive mode 成功的关键函数
# plt.figure(1)
# t = [0]
# t_now = 0
# m = [sin(t_now)]

# for i in range(2000):
#     plt.clf() #清空画布上的所有内容
#     t_now = i*0.1
#     t.append(t_now)#模拟数据增量流入，保存历史数据
#     m.append(sin(t_now))#模拟数据增量流入，保存历史数据
#     plt.plot(t,m,'-r')
#     plt.pause(0.01)

# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
##########scipy 凸包################
# points = np.random.rand(30, 2)
points = np.array([[680.,489.],
 [675.32088886,501.85575219],
 [663.47296355,508.69615506],
 [650.        ,506.32050808],
 [641.20614758,495.84040287],
 [641.20614758,482.15959713],
 [650.        ,471.67949192],
 [663.47296355,469.30384494],
 [675.32088886,476.14424781],
 [680.        ,489.        ],
 [700.        ,455.        ],
 [695.32088886,467.85575219],
 [683.47296355,474.69615506],
 [670.        ,472.32050808],
 [661.20614758,461.84040287],
 [661.20614758,448.15959713],
 [670.        ,437.67949192],
 [683.47296355,435.30384494],
 [695.32088886,442.14424781],
 [700.        ,455.        ]])
# print(points)
hull = ConvexHull(points)
# plt.plot(points[:,0], points[:,1], 'o')
# hull.vertices 得到凸轮廓坐标的索引值，逆时针画
hull1=hull.vertices.tolist()#要闭合必须再回到起点[0]
hull1.append(hull1[0])
plt.plot(points[hull1,0], points[hull1,1], 'r--',lw=2)
# for i in range(len(hull1)-1):
#     plt.text(points[hull1[i],0], points[hull1[i],1],str(i),fontsize=20)
plt.show()
