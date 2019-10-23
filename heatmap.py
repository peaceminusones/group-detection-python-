"""
    paths convergence |  feature_pc  |  heatmap
"""

import numpy as np 
import math
import copy

def heatmap(traj1, traj2, video_par):

    # units of measurements
    m = 1
    cm = m/100
    cellSide = 30*cm

    C = 1
    k_p = 0.5
    k_t= 0.000001

    xMin = video_par['xMin']
    xMax = video_par['xMax']
    yMin = video_par['yMin']
    yMax = video_par['yMax']
    numberofCellForSide_1 = math.floor((xMax - xMin)/cellSide)
    numberofCellForSide_2 = math.floor((yMax - yMin)/cellSide)

    # matrices for traj1
    heat_1 = np.zeros((numberofCellForSide_2, numberofCellForSide_1))
    H_1 = copy.deepcopy(heat_1)
    gridStart_1 = np.ones((numberofCellForSide_2, numberofCellForSide_1))*np.nan
    gridEnd_1 = copy.deepcopy(gridStart_1)

    # matrices for traj2
    heat_2 = np.zeros((numberofCellForSide_2, numberofCellForSide_1))
    H_2 = copy.deepcopy(heat_2)
    gridStart_2 = np.ones((numberofCellForSide_2, numberofCellForSide_1))*np.nan
    gridEnd_2 = copy.deepcopy(gridStart_2)

    # define the length of the analysis
    frame_start = int(min(traj1[0,0], traj2[0,0]))
    frame_end = int(max(traj1[-1,0], traj2[-1,0]))
    step = video_par['downsampling']
    for i in range(frame_start, frame_end + 1, step):
        # trace the path of the trajectory of traj1
        if i in traj1[:, 0]:
            index = list(traj1[:,0]).index(i)
            grid_1 = min(max(math.floor(traj1[index, 1]/cellSide), 1), numberofCellForSide_1)
            grid_2 = min(max(math.floor(traj1[index, 2]/cellSide), 1), numberofCellForSide_2)
            if math.isnan(gridStart_1[-grid_2, grid_1-1]):
                gridStart_1[grid_2-1, grid_1-1] = i
                gridEnd_1[grid_2-1, grid_1-1] = i
            else:
                gridEnd_1[grid_2-1, grid_1-1] = i

        # trace the path of the trajectory of traj2
        if i in traj2[:, 0]:
            index = list(traj2[:,0]).index(i)
            grid_1 = min(max(math.floor(traj2[index, 1]/cellSide), 1), numberofCellForSide_1)
            grid_2 = min(max(math.floor(traj2[index, 2]/cellSide), 1), numberofCellForSide_2)
            if math.isnan(gridStart_2[-grid_2, grid_1-1]):
                gridStart_2[grid_2-1, grid_1-1] = i
                gridEnd_2[grid_2-1, grid_1-1] = i
            else:
                gridEnd_2[grid_2-1, grid_1-1] = i

    # print(gridStart_1)
    # print(gridEnd_1)
    # print(gridStart_2)
    # print(gridEnd_2)

    return H, S