# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 20:57:00 2020
@author: cheng,tang,liao
"""
import numpy as np
import matplotlib.pyplot as plt
from group_detection import get_prediction

def circle_group_grid(offsets, data, size, dist_thre=1.5, ratio=0.9, max_friends=100):
    '''
    This function computes circular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        offsets: num_trajs*seq_length*8; 8 refers to [frameId, userId, x, y, delta_x, delta_y, heading, velocity]
    data: trajectory data with all the frames and all the users, [frameId, userId, x, y]
    size: the height and width of the occupancy
    '''
    group_model_input = []
        
    # Get the friends data
    friends = get_prediction(data, dist_thre=dist_thre, ratio=ratio, max_friends=max_friends)
            
    for ego_data in offsets:
        o_map = get_circle_occupancy_map(ego_data, offsets, size, friends)
        group_model_input.append(o_map)
    group_model_input = np.reshape(group_model_input, 
                                   [len(group_model_input), len(ego_data), size[0], size[1], 3])
    return group_model_input
def get_circle_occupancy_map(ego_data, offsets, size, friends=None, islog=False, max_speed=6):
    '''
    This is the function to get the occupancy for each ego user
    '''
    # the size of the anggle and radius
    height, width = int(size[0]), int(size[1])
    o_map = np.zeros((len(ego_data), height, width, 3))    
    offsets = np.reshape(offsets, (-1, 8))
    offsets[:, -1] = np.clip(offsets[:, -1], a_min=0, a_max=max_speed)    
    pad = 0
    egoId = ego_data[0, 1]
    egoFrameList = ego_data[:, 0]
    # Get the ego user's friends
    ego_friends = friends[friends[:, 0]==egoId, :]
    ## comment out this to show the dynamic maps
    ## Note, it will slow down the data processing remarkably
    # plt.ion()    
    for i, f in enumerate(egoFrameList):
        count = np.zeros((height, width))
        speed_map = np.zeros((height, width))
        orient_map = np.zeros((height, width))
        
        frame_data = offsets[offsets[:, 0]==f, :]
        otherIds = frame_data[:, 1]
        current_x, current_y = ego_data[i, 2], ego_data[i, 3]
        current_delty_x, current_delty_y = ego_data[i, 4], ego_data[i, 5]
        
        for otherId in otherIds:
            if egoId != otherId:
                ### incorporate friend detection
                if otherId in ego_friends:
                    # print('%s and %s are frineds'%(str(int(egoId)), str(int(otherId))))
                    # # Set the occupancy as 0
                    continue
                
                [other_x, other_y, other_delty_x, other_delty_y, 
                 other_theata, other_velocity] = frame_data[frame_data[:, 1]==otherId, 2:][0]
                                
                # Get the relative position of the current use to the ego user
                # Put the postion of the ego user in the centriod
                delta_h = int(np.floor(other_y-current_y) + height/2)
                delta_w = int(np.floor(other_x-current_x) + width/2)
                                
                # only in the vicinity 
                if delta_h<=height and delta_w<=width and delta_h>=0 and delta_w>=0:
                    
                    # Compute the relative movement in x
                    xl, xr = sorted((delta_w, delta_w+other_delty_x-current_delty_x))
                    xl= max(int(np.floor(xl)-pad), 0)
                    xr = min(int(np.floor(xr)+pad), height)
                    # Compute the relative movement in y
                    yl, yr = sorted((delta_h, delta_h+other_delty_y-current_delty_y))
                    yl= max(int(np.floor(yl)-pad), 0)
                    yr = min(int(np.floor(yr)+pad), height)

                    # count the frequency for normalization
                    count[yl:yr, xl:xr] += 1
                    
                    # Normalize to [0, 1]                    
                    speed_map[yl:yr, xl:xr] += other_velocity/max_speed # the maximum speed is set to 6 m/s
                    orient_map[yl:yr, xl:xr] += (other_theata+np.pi)/(2*np.pi)
        
        # Normalize the aggregated occupancy
        orient_map = normalize(count, orient_map)       
        speed_map = normalize(count, speed_map)
        if np.max(count)!=0:
            count = count/np.max(count) 
        
        o_map[i, :, :, 0] = orient_map
        o_map[i, :, :, 1] = speed_map                               
        o_map[i, :, :, 2] = count
                   
        assert np.max(o_map[i])<=1 and np.min(o_map[i])>=0,\
            print("Occupancy not normalized", np.max(o_map[i]), np.min(o_map[i]), np.max(offsets[:, -1]))
    return o_map
def normalize(count, data):
            norm_data = np.zeros_like(data)
            for i, row in enumerate(data):
                for j, value in enumerate(row):
                    if count[i, j] != 0:
                        norm_data[i, j] = value/count[i, j]
            return norm_data