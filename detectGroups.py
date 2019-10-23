"""
    detect groups:
"""

import numpy as np

def detectGroups(couples, allHeatMaps):
    """
    数据结构：
        couples：二维矩阵
        allHeatMaps：dict(), {i: heatmap} 表示第i对couple的热图
        其中len(allHeatMaps)为1128表示有1128个couple
        allHeatMaps[i]表示第i对couple的热图，大小为(56,116)
    """
    detectedGroups = dict()
    # 累计的HM值
    cumulativeHM = allHeatMaps[0]
    for i in range(1, len(allHeatMaps)):
        cumulativeHM = cumulativeHM + allHeatMaps[i]

    # normalize
    # 且cumulativeHM从(56,116)转置为(116,56)
    cumulativeHM = np.transpose(cumulativeHM) / (np.max(cumulativeHM))  # 矩阵的转置除以矩阵的最大值

    """
        聚类算法初始化
    """
    convergingPos = np.zeros((len(allHeatMaps), 2))  # convergingPos合并位置，大小：（1128，2）
    for i in range(len(allHeatMaps)):
        # 判断每个couple是否有成组的意义
        if np.max(allHeatMaps[i]) < 0.0001: 
            convergingPos[i, :] = [-1, -1]
            continue
        
        # 如果有成组的意义:
        # 先找到allheatmap[i]中第一个等于np.max(allheatmap[i])的位置
        # 即最有可能是couple的第一个位置
        index = np.argwhere(allHeatMaps[i] == np.max(allHeatMaps[i]))
        row = index[0, 0]
        column = index[0, 1]
        convergingPos[i, :] = [row, column]

    # 然后从第一个位置开始朝着cumulativeHM定义的分布的局部最大值移动
    """
        梯度上升聚类
    """
    # 为了便于处理边界，把cumulativeHM矩阵都扩大2
    cumulativeHM_augmented = np.zeros((cumulativeHM.shape[0] + 2, cumulativeHM.shape[1]+2))
    cumulativeHM_augmented[2:-2, 2:-2]= cumulativeHM

    changed = np.ones(len(allHeatMaps), 1)
    while sum(changed) > 1:
        for i in range(len(allHeatMaps)):
            if changed[i] == 1 and convergingPos[i,:] == [-1, -1]:
                changed[i] = 0
            else:
                changed[i] = 0
                continue
        

    return detectedGroups