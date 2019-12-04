
"""
    1) 检查是否只检测到一个组，如果只检查到一个组，则任意两个cluster可以合并
    2) 检查每一个可能的元素组合是否构成一个现有的couple
    3) 检查这个group是否是两个singleton：如果是的话，couple是存在的，是合法的
    4) 至少有一个cluster中包含有多个元素。一定会选择这样的cluster，因为它一定会提供我们cluster中couple的位置，而我们可以更好的修复为正确的
"""
import numpy as np

def isClusterLegal(cluster1, cluster2, detectedGroups):
    if len(detectedGroups) == 1:
        return True
    
    return False
    