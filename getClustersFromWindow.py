"""
    从‘clusters.txt’文件中读取当前窗口下存在的group，用于训练
"""

import numpy as np

def getClustersFromWindow(trackid, dataDirectory):
    clusters = []
    fid = open(dataDirectory + '/clusters.txt', 'r')
    line = fid.readline()
    while line:
        line_split = line[:-1].split() # 去掉换行符
        line_intlist = intlist(line_split)
        # 判断每个群组元素是否在当前的窗口出现
        join = list(set(line_intlist) & set(trackid))
        if join:#set(line_intlist) <= set(trackid):
            if len(join) > 1:
                join = [[join[i]] for i in range(len(join))]
            clusters.append(join) 
        line = fid.readline()
    fid.close() #关闭文件
    # print(clusters)
    return clusters

# 把list里数据的数据结构从str转为int
def intlist(line_split):
    for i in range(len(line_split)):
        line_split[i] = int(line_split[i])
    return line_split