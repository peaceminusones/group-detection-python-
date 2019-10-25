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

    for i in range(heat_1.shape[0]):
        for j in range(heat_1.shape[1]):
            if not math.isnan(gridStart_1[i,j]):
                Ebar = C/k_t * (1 - math.exp(-k_t * (gridEnd_1[i,j] - gridStart_1[i,j] + 1)/video_par['downsampling']))
                heat_1[i,j] = Ebar * math.exp(-k_t *(frame_end - gridEnd_1[i,j] + 1)/video_par['downsampling'])
            
            if not math.isnan(gridStart_2[i,j]):
                Ebar = C/k_t * (1 - math.exp(-k_t * (gridEnd_2[i,j] - gridStart_2[i,j] + 1)/video_par['downsampling']))
                heat_2[i,j] = Ebar * math.exp(-k_t *(frame_end - gridEnd_2[i,j] + 1)/video_par['downsampling'])
    
    # interesting patches
    z_1 = copy.deepcopy(heat_1)
    z_1[z_1!=0] = 1
    N_1 = sum(sum(z_1))
    z_2 = copy.deepcopy(heat_2)
    z_2[z_2!=0] = 1
    N_2 = sum(sum(z_2))

    for i in range(heat_1.shape[0]):
        for j in range(heat_1.shape[1]):
            # now compute the diffusion for traj1
            for m in range(z_1.shape[0]):
                for n in range(z_1.shape[1]):
                    dist = math.sqrt(math.pow(i-m,2) + math.pow(j-n,2))
                    H_1[i,j] = H_1[i,j] + heat_1[m,n] * math.exp(-k_p*dist)

            # compute the diffusion for traj2
            for m in range(z_2.shape[0]):
                for n in range(z_2.shape[1]):
                    dist = math.sqrt(math.pow(i-m,2) + math.pow(j-n,2))
                    H_2[i,j] = H_2[i,j] + heat_2[m,n] * math.exp(-k_p*dist)

    H_1 = H_1/N_1
    sum_1 = sum(sum(H_1))

    H_2 = H_2/N_2
    sum_2 = sum(sum(H_2))

    H = H_1 * H_2
    sum_H = sum(sum(H))

    if min(sum_1, sum_2) == 0:
        S = 0
    else:
        S = sum_H / min(sum_1, sum_2)

    return H, S