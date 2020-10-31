# -*- coding: utf-8 -*-

import numpy as np
from ranking import gauss_rank
from maps import Maps


def test():
    path = "../WORLD H-H TRAJ/Train/crowds/crowds_zara03.txt"
    trajs = sort_data(path, 20)
    trajs = trajs[:, :8, :]
    print(trajs.shape)


def sort_data(path, seq_length):
    data = np.genfromtxt(path, delimiter='')
    # challenge dataset have nan for prediction time steps        
    data = data[~np.isnan(data).any(axis=1)]
    datamaps = Maps(data)
    trajs = np.reshape(datamaps.sorted_data, (-1, seq_length, 4))
    return trajs


def write_pred_txt(obs_trajs, pred_trajs, dataname, folder):     
    with open("../%s/%s.txt"%(folder, dataname), 'w+') as myfile:
        for i, obs_traj in enumerate(obs_trajs):
            start_frame, pedestrian, _, _ = obs_traj[0]
            step = obs_traj[-1, 0] - obs_traj[-2, 0]
            predictions = pred_trajs[i] 
            
            if predictions.shape[0]>1: 
                # Get the ranks and select the most likely prediction
                ranks = gauss_rank(predictions)
                ranked_predicion = predictions[np.argmax(ranks)]
            else:
                ranked_predicion = predictions[0]
                
            traj = np.concatenate((obs_traj[:, 2:4], ranked_predicion), axis=0)
            for j, pos in enumerate(traj): 
                myfile.write(str(int(start_frame+j*step))+' '+
                             str(round(pedestrian, 0))+' '+
                             str(round(pos[0], 3))+' '+
                             str(round(pos[1], 3))+' '+'\n')
    myfile.close()
    
    
def get_index(obs_trajs, pred_trajs):
    
    trajectories = []
    
    for i, obs_traj in enumerate(obs_trajs):
        start_frame, pedestrian, _, _ = obs_traj[0]
        step = obs_traj[-1, 0] - obs_traj[-2, 0]
        predictions = pred_trajs[i] 
        
        if predictions.shape[0]>1:        
            # Get the ranks and select the most likely prediction
            ranks = gauss_rank(predictions)
            ranked_predicion = predictions[np.argmax(ranks)]
        else:
            ranked_predicion = predictions[0]
            
        traj = np.concatenate((obs_traj[:, 2:4], ranked_predicion), axis=0)
        for j, pos in enumerate(traj):
            trajectories.append([start_frame+j*step,
                                 pedestrian,
                                 pos[0],
                                 pos[1]])

    return np.asarray(trajectories)
if __name__ == "__main__":
    test()