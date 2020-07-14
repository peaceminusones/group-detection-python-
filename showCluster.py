
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy.spatial import ConvexHull

def showCluster(myF, Y_test, Y_pred, model_par, video_par, dataDirectory):

    for i in range(len(Y_test)):
        Y_test[i] = Y_test[i]

    video_par['xyReversed'] = video_par['xyReversed'] if video_par['xyReversed'] == 1 else 0

    # this variable add a subplot if we also have a video to display
    n_subplots = 4
    index_start = 0
    index_end= index_start
    colors = 'rgmcyb'
    allpath = dict()
    # # GUI----------------------------------
    # root = Tk()

    # # -------------------------------------
    plt.figure(figsize=(14,8))
    for j in range(model_par.testingSetSize + model_par.trainingSetSize):

        while (index_end <= myF.shape[0]) and (myF.iloc[index_end, 0] <= (myF.iloc[index_start, 0] + model_par.window_size * video_par['frame_rate'])):
            index_end = index_end + 1
        # 需要把10s内所有帧的数组提出来
        myF_scene = myF.iloc[index_start: index_end, :]
        # # 从10s的第一帧到最后一帧，把每个人的轨迹提取出来，其中数据类型{int_path_id: dataframe_path}
        # path = dict()
        # F = myF_scene.values
        # # 将10s内所有帧的不重复的track_id提取出来，表示有多少个人
        # track_id = sorted(set(F[:,1].astype(int)))
        # for t in track_id:
        #     choosebytrackid = myF_scene[myF_scene[1] == t]
        #     path[t] = choosebytrackid.iloc[:,[0,2,4]].values
        
        path = dict()
        path[1] = []
        start_frame = myF_scene.iloc[0,0]
        end_frame = myF_scene.iloc[-1,0]
        plt.clf()
        for f in range(start_frame, end_frame + 1):
            # 把帧f下的所有信息提取出来
            choosebyframeid = myF_scene[myF_scene[0] == f]
            # 把行人的trackid单独提取出来
            pedestrain = choosebyframeid.iloc[:,1].values
            # 相应的位置信息提取出来
            locations = choosebyframeid.iloc[:,[2,4]].values
            h1 = []
            h2 = []
            for i in range(len(pedestrain)):
                if 1 or (pedestrain[i] in Y_test[j]):
                    if pedestrain[i] not in path.keys():
                        path[pedestrain[i]] = []
                    # print(len(path))
                    # print(pedestrain[i])
                    # print(path[pedestrain[i]])
                    # print([f] + list(locations[i,:]))
                    path[pedestrain[i]].append([f] + list(locations[i,:]))

                    index_found_ground = -1
                    index_found_predicted = -1
                    
                    for p in range(len(Y_test[j])):
                        if pedestrain[i] in Y_test[j][p]:
                            index_found_ground = p
                    for p in range(len(Y_pred[j])):
                        if pedestrain[i] in Y_pred[j][p]:
                            index_found_predicted = p
                    
                    # plt.subplot(221)
                    # if index_found_ground != -1:
                    #     if video_par['xyReversed']:
                    #         ppi1 = np.array(path[pedestrain[i]])[:,1]
                    #         ppi2 = np.array(path[pedestrain[i]])[:,2]
                    #         # print("yes")
                    #         # print(ppi1)
                    #         # print(ppi2)
                    #         # im = cv2.imread("D:\\group detection and prediction\\group-detection-python\\mydata\\student003" + '\\' + SixNumber(f) + '.jpg')
                    #         # plt.imshow(im)
                    #         c = colors[index_found_ground % len(colors)]
                    #         plt.plot(ppi1, ppi2, '-', color = c)
                    #         h1.append(plt.text(ppi1[-1], ppi2[-1], str(pedestrain[i])))
                    #     else:
                    #         ppi1 = np.array(path[pedestrain[i]])[:,2]
                    #         ppi2 = np.array(path[pedestrain[i]])[:,1]
                    #         # print("no")
                    #         # print(ppi1)
                    #         # print(ppi2)
                    #         c = colors[index_found_ground % len(colors)]
                    #         plt.plot(ppi1, ppi2, '-', color = c)
                    #         h1.append(plt.text(ppi1[-1], ppi2[-1], str(pedestrain[i])))
                        
                    # if video_par['xyReversed']:
                    #     plt.axis([video_par['yMin'], video_par['yMax'], video_par['xMin'], video_par['xMax']])
                    # else:
                    #     plt.axis([video_par['xMin'], video_par['xMax'], video_par['yMin'], video_par['yMax']])

                    # plt.title('Ground truth')
                    # if video_par['isYreversed']:
                    #     plt.gca().invert_yaxis()

                    # plt.subplot(222)
                    # if index_found_predicted != -1:
                    #     if len(Y_pred[j][index_found_predicted]) < 2:
                    #         if video_par['xyReversed']:
                    #             ppi1 = np.array(path[pedestrain[i]])[:,1]
                    #             ppi2 = np.array(path[pedestrain[i]])[:,2]
                    #             c = colors[index_found_ground % len(colors)]
                    #             plt.plot(ppi1, ppi2, 'k')
                    #         else:
                    #             ppi1 = np.array(path[pedestrain[i]])[:,2]
                    #             ppi2 = np.array(path[pedestrain[i]])[:,1]
                    #             c = colors[index_found_ground % len(colors)]
                    #             plt.plot(ppi1, ppi2, 'k')
                    #     else:
                    #         if video_par['xyReversed']:
                    #             ppi1 = np.array(path[pedestrain[i]])[:,1]
                    #             ppi2 = np.array(path[pedestrain[i]])[:,2]
                    #             c = colors[index_found_ground % len(colors)]
                    #             plt.plot(ppi1, ppi2, color = c)
                    #             h2.append(plt.text(ppi1[-1], ppi2[-1], str(pedestrain[i])))
                    #         else:
                    #             ppi1 = np.array(path[pedestrain[i]])[:,2]
                    #             ppi2 = np.array(path[pedestrain[i]])[:,1]
                    #             c = colors[index_found_ground % len(colors)]
                    #             plt.plot(ppi1, ppi2, color = c)
                    #             h2.append(plt.text(ppi1[-1], ppi2[-1], str(pedestrain[i])))
                    #     if video_par['xyReversed']:
                    #         plt.axis([video_par['yMin'], video_par['yMax'], video_par['xMin'], video_par['xMax']])
                    #     else:
                    #         plt.axis([video_par['xMin'], video_par['xMax'], video_par['yMin'], video_par['yMax']])

                    #     plt.title('Predicted clustering')
                    #     if video_par['isYreversed']:
                    #         plt.gca().invert_yaxis()

            circle = []
            if video_par['videoObj'] and (f % video_par['downsampling'] == start_frame % video_par['downsampling']):
                t = np.linspace(0, 2*math.pi, 10)
                r = 20
                myvideo = cv2.imread(video_par['videoObj']%f)
                sp = myvideo.shape
                # plt.subplot(223)
                # plt.cla()
                # plt.imshow(myvideo)
                # for p in range(len(Y_test[j])):
                #     if len(Y_test[j][p]) > 1:
                #         cluster_points = np.zeros(2)
                #         for q in range(len(Y_test[j][p])):
                #             if Y_test[j][p][q] in pedestrain:
                #                 pedestrain_id = Y_test[j][p][q][0]
                #                 data = np.array([[path[pedestrain_id][-1][1], path[pedestrain_id][-1][2]]])
                #                 data = np.append(np.array(data), np.ones((data.shape[0], 1)))
                #                 data = np.dot(video_par['H'], data)
                #                 repmat = np.array([data[2] for _ in range(3)])
                #                 data = np.round(data/repmat)
                                
                #                 x = np.array([r*math.cos(t[i]) + data[0] for i in range(len(t))]).reshape(-1,1)
                #                 y = np.array([r*math.sin(t[i]) + data[1] for i in range(len(t))]).reshape(-1,1)
                #                 comb = np.hstack((x,y))
                #                 cluster_points = np.vstack((cluster_points, comb))
                #         cluster_points = np.delete(cluster_points, 0, axis=0)
                #         # print("before --------------------------")
                #         # print(cluster_points)
                #         if cluster_points.shape[0] > 0 and cluster_points.shape[0] > 1:
                #             # print("after ----------------------------")
                #             # print(cluster_points)
                #             k = ConvexHull(cluster_points)
                #             k1 = k.vertices.tolist()
                #             k1.append(k1[0]) #要闭合必须再回到起点[0]
                #             # print(k1)
                #             c = colors[p % len(colors)]
                #             circle = plt.plot(cluster_points[k1,0], cluster_points[k1,1], color = c, alpha=1)

                plt.subplot(111)
                plt.cla()
                plt.imshow(myvideo, shape=sp)
                for p in range(len(Y_pred[j])):
                    if len(Y_pred[j][p]) > 1:
                        cluster_points = np.zeros(2)
                        for q in range(len(Y_pred[j][p])):
                            if Y_pred[j][p][q] in pedestrain:
                                pedestrain_id = Y_pred[j][p][q][0]
                                data = np.array([[path[pedestrain_id][-1][1], path[pedestrain_id][-1][2]]])
                                data = np.append(np.array(data), np.ones((data.shape[0], 1)))
                                data = np.dot(video_par['H'], data)
                                repmat = np.array([data[2] for _ in range(3)])
                                data = np.round(data/repmat)
                                
                                x = np.array([r*math.cos(t[i]) + data[0] for i in range(len(t))]).reshape(-1,1)
                                y = np.array([r*math.sin(t[i]) + data[1] for i in range(len(t))]).reshape(-1,1)
                                comb = np.hstack((x,y))
                                cluster_points = np.vstack((cluster_points, comb))
                        cluster_points = np.delete(cluster_points, 0, axis=0)
                        # print("before --------------------------")
                        # print(cluster_points)
                        if cluster_points.shape[0] > 0 and cluster_points.shape[0] > 1:
                            # print("after ----------------------------")
                            # print(cluster_points)
                            k = ConvexHull(cluster_points)
                            k1 = k.vertices.tolist()
                            k1.append(k1[0]) #要闭合必须再回到起点[0]
                            # print(k1)
                            c = colors[p % len(colors)]
                            circle = plt.plot(cluster_points[k1,0], cluster_points[k1,1], color = c, alpha=1)
            
            plt.axis('off')
            plt.draw()
            plt.pause(0.01)
            plt.savefig("D:/pedestrian analysis/group detection and prediction/group-detection-python/frame/frame_" + str(f) + ".png")
            
            for i in range(len(h1)):
                plt.Text.set_visible(h1[i], b = False)
            for i in range(len(h2)):
                plt.Text.set_visible(h2[i], b = False)
    
        allpath[j] = path
        index_start = index_end + 1
        index_end = index_start    
    
    return

def SixNumber(str_number):
    str_number=str(str_number)
    while(len(str_number)<6):
        str_number='0'+str_number
    return str_number