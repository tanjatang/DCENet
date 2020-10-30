# -*- coding: utf-8 -*-


import numpy as np
import writer
from maps import Maps
    

def check_collision(data, thred=0.1, n=1, obs_seq=8, pred_seq=12):
    """
    data: frameId, pedId, x, y
    data.shape: num_trajs*seq_length*4, 
    """
    print("The shape of the input data", data.shape)
    if len((data.shape))==2:
        datamaps = Maps(data)
        data = np.reshape(datamaps.sorted_data, (-1, obs_seq+pred_seq, 4))
        data = data [:, obs_seq:, :]
        print("The shape of the new data", data.shape)
    
    
    count_collisions = 0
    encounters = 0
    traj_data = data.reshape(-1, 4)
    
    for ped_traj in data:
        
        ego_pedid = ped_traj[0, 1]
        ped_frameIds = ped_traj[:, 0]
                
        co_traj_data = traj_data[traj_data[:, 0]>=np.min(ped_frameIds)]
        co_traj_data = co_traj_data[co_traj_data[:, 0]<=np.max(ped_frameIds)]
        co_pedids = np.unique(co_traj_data[:, 1])
        
        for co_pedid in co_pedids:
            if co_pedid != ego_pedid:
                con_ped_traj = co_traj_data[co_traj_data[:, 1]==co_pedid]
                if con_ped_traj.size != 0:
                    encounters += 1
                    count = count_collision(ped_traj, con_ped_traj, thred, n)
                    count_collisions += count
                    
    print("Total trajectories %.0f, Total encounters %.0f, collisions %.0f, collision rate %.2f"%
          (len(data), encounters, count_collisions, count_collisions/encounters))                    
    return encounters, count_collisions
    


def count_collision(ego_ped_traj, co_ped_traj, thred, n):
    ego_ped_traj = ego_ped_traj[ego_ped_traj[:, 0]>=np.min(co_ped_traj[:, 0])]
    ego_ped_traj = ego_ped_traj[ego_ped_traj[:, 0]<=np.max(co_ped_traj[:, 0])]
      
    # Interpolation
    ego_x, ego_y = linear_interp(ego_ped_traj[:, 2], ego_ped_traj[:, 3], n)
    co_x, co_y = linear_interp(co_ped_traj[:, 2], co_ped_traj[:, 3], n)
        
    count = eucl_dis(np.vstack((ego_x, ego_y)).T, np.vstack((co_x, co_y)).T, n, thred) 
    return count
    
    
def linear_interp(x, y, n):
    '''
    This is the one-D linear interpolation in each dimension
    '''
    xvals_ = np.empty((0))
    yvals_ = np.empty((0))
    for i, j in enumerate(x):
        if i<len(x)-1:
            x_ = [k*(x[i+1] - j)/n + j for k in range(n)]
            xvals_ = np.hstack((xvals_, x_))
            y_ = [k*(y[i+1] - y[i])/n + y[i] for k in range(n)]
            yvals_ = np.hstack((yvals_, y_))                        
    xvals_ = np.hstack((xvals_, x[-1]))
    yvals_ = np.hstack((yvals_, y[-1]))

    
    # Numpy linear interpolation
    # This works the same as yvals_
    # yinterp = np.interp(xvals_, x, y)    
    # plt.plot(xvals_, yinterp, '-*', color='blue')
   
    return xvals_, yvals_
    
def eucl_dis(coords1, coords2, n=1, thred=0.001):
    '''
    This is the function to claculate the time-aligned euclidean distance for two trajectories
    n: The number of interpolated points in two adjacent points
    thred: the minimum distance for collision
    '''
    count = 0
    dist = np.linalg.norm((coords1-coords2), axis=1)
    if np.any(dist<thred):
        pos = np.where(dist<thred)[0]
        time_pos = pos/n
        # print("collision in prediction at time", time_pos)
        count += 1
    return count
