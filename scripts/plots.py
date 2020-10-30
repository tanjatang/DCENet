# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:49:28 2020

@author: cheng
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

def plot_fov(traj, rois, enviro, name=None):   
    fig,ax = plt.subplots()
    
    enviro = enviro*255
    [index] = np.random.randint(len(rois), size=1)
    # print(index)
    # 212, 195
    # index= 212
    # print(np.unique(enviro))
    ax.imshow(enviro.astype(int))
    # plt.imshow(environ.astype(int))
    for i, roi in enumerate(rois[index]):
        xl, yl, xr, yr = roi
        xl, yl = int(xl*enviro.shape[1]), int(yl*enviro.shape[0])
        w, h = int(xr*enviro.shape[1]-xl), int(yr*enviro.shape[0]-yl)
        rect = patches.Rectangle((xl,yl),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # ax.text(traj[index, i, 0], traj[index, i, 1], str(int(i+1)), fontsize=10)
    ax.plot(traj[index, :, 0], traj[index, :, 1], color='w', marker='s', markersize=2)
    # plt.savefig("../plot_environment/%s_field_of_view.png"%name, bbox_inches="tight", dpi=350)
    plt.show()
    

def plot_environ(enviro, name=None):
    extent = [0, enviro.shape[1], enviro.shape[0], 0]
    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")   
    ax2.set_aspect("equal") 
    ax0.imshow(enviro[:, :, 0], extent=extent, origin='upper', cmap=cm.jet, vmin=0, vmax=1)
    ax1.imshow(enviro[:, :, 1], extent=extent, origin='upper', cmap=cm.jet, vmin=0, vmax=1)
    ax2.imshow(enviro[:, :, 2], extent=extent, origin='upper', cmap=cm.jet, vmin=0, vmax=1) 
    ax0.set_title("Trajectory")
    ax1.set_title("Orientaion")
    ax2.set_title("Speed")      
    # plt.colorbar(img, ax=ax2)
    # plt.savefig("../plot_environment/%s_enviroment.png"%name, bbox_inches="tight", dpi=350)
    plt.show()
    plt.gcf().clear()
    plt.close()    
  

def plot_maps(map, img_extent, des, dataname=None):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")    
    img = ax.imshow(map, extent=img_extent, origin='upper', cmap=cm.jet)
    plt.colorbar(img, ax=ax)
    ax.set_title(des)
    # plt.savefig("fig/test_real/%s_speed_map.png"%(dataname), bbox_inches='tight', dpi=300)
    plt.show()
    plt.gcf().clear()
    plt.close("all")
    
    
def plot_pred(xy, y_prime, N=10, groundtruth=True):
    """
    This is the plot function to plot the first scene
    """
    
    fig,ax = plt.subplots()
    pred_seq = y_prime.shape[2]
    obs_seq = xy.shape[1] - pred_seq
    
    if groundtruth:
        for i in range(N):
            # plot observation
            ax.plot(xy[i, :obs_seq, 2], xy[i, :obs_seq, 3], color='k')
            # plot ground truth
            ax.plot(xy[i, obs_seq-1:, 2], xy[i, obs_seq-1:, 3], color='r')
            for j, pred in enumerate(y_prime[i]):
                # concate the first step for visulization purpose
                pred = np.concatenate((xy[i, obs_seq-1:obs_seq, 2:4], pred), axis=0)            
                ax.plot(pred[:, 0], pred[:, 1], color='b')                
    else:
        x = xy
        obs_seq = x.shape[1]        
        for i in range(N):
            # plot observation
            ax.plot(x[i, :, 2], x[i, :, 3], color='k')
            for j, pred in enumerate(y_prime[i]):
                # concate the first step for visulization
                pred = np.concatenate((x[i, obs_seq-1:obs_seq, 2:4], pred), axis=0)            
                ax.plot(pred[:, 0], pred[:, 1], color='b')                
    ax.set_aspect("equal")
    plt.show()
    plt.gcf().clear()
    plt.close()        