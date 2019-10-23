
import numpy as np
import matplotlib.pyplot as plt


def showCluster(myF, Y_test, Y_pred, model_par, video_par):

    for i in range(len(Y_test)):
        Y_test[i] = Y_test[str(i)]

    video_par['xyReversed'] = video_par['xyReversed'] if video_par['xyReversed'] == 1 else 0

    # this variable add a subplot if we also have a video to display
    n_subplots = 4
    index_start = 0
    index_end= index_start
    colors = 'rgbmcykw'
    plt.figure(figsize=(14,8))

    for j in range(model_par.testingSetSize):

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
        for f in range(start_frame, end_frame + 1):
            # 把帧f下的所有信息提取出来
            choosebyframeid = myF_scene[myF_scene[0] == f]
            # 把行人的trackid单独提取出来
            pedestrain = choosebyframeid.iloc[:,1].values
            # 相应的位置信息提取出来
            locations = choosebyframeid.iloc[:,[2,4]].values
            
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
                    
                    plt.subplot(221)
                    if index_found_ground != -1:
                        if video_par['xyReversed']:
                            # print(np.array(path[pedestrain[i]])[:,1])
                            # print(np.array(path[pedestrain[i]])[:,2])
                            ppi1 = np.array(path[pedestrain[i]])[:,1]
                            ppi2 = np.array(path[pedestrain[i]])[:,2]
                            c = colors[index_found_ground % 8]
                            plt.plot(ppi1, ppi2, 'r--', color = c)
                            plt.text(ppi1[-1], ppi2[-1], str(pedestrain[i]))
                        else:
                            ppi1 = np.array(path[pedestrain[i]])[:,2]
                            ppi2 = np.array(path[pedestrain[i]])[:,1]
                            c = colors[index_found_ground % 8]
                            plt.plot(ppi1, ppi2, 'r--', color = c)
                            plt.text(ppi1[-1], ppi2[-1], str(pedestrain[i]))
                    
                    if video_par['xyReversed']:
                        plt.axis([video_par['yMin'], video_par['yMax'], video_par['xMin'], video_par['xMax']])
                    else:
                        plt.axis([video_par['xMin'], video_par['xMax'], video_par['yMin'], video_par['yMax']])

                    plt.title('Ground truth')
                    if video_par['isYreversed']:
                        plt.gca().invert_yaxis()


        # return 
    return