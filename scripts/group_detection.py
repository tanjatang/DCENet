# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:40:15 2018
This is the module to extract the road users coexisting with a given ego user
@author: cheng
"""
import numpy as np
from sklearn.cluster import DBSCAN

#from group_evaluation import get_IoU

def get_prediction(sequence, dist_thre=1.5, ratio=0.90, max_friends=100):
    '''
    Extract ego user's using group_detection
    '''   
    Detector = Group_Detection(data=sequence, dist_thre=dist_thre, ratio_thre=ratio)      
    # Define the largest number of friends an ego user can have (here, include the ego user self)
    # This number must be large enough to harvest all possibilities
    t_friends = np.zeros([Detector.userList.shape[-1], max_friends])          
    for count, egoUserId in enumerate(Detector.userList):
        userData = Detector.data[Detector.data[:, 1]==egoUserId, :]
        if egoUserId != 0:
            egoUserFl = np.unique(userData[:, 0])          
            frameData = Detector.get_frame_data(egoUserFl)
            friends = Detector.frame_DBscan(frameData, egoUserId, egoUserFl)                        
            store_fl = np.append([egoUserId], friends)
            t_friends[count, 0:store_fl.shape[-1]] = store_fl                  
    return t_friends


class Group_Detection():
    '''
    This is the class for group detection, which is a time sequence DBSCAN:
        DBSCAN_friend: Using DBSCAN to cluster friends into group based on Euclidean distance
    '''
    def __init__(self, data, dist_thre=3, ratio_thre=0.9):
        '''
        params:
            data_dir: it is the place where trajectory data resident
            dist_thre: Euclidean distance threshold for defining a friend
            ratio_thre: overlap threshold for defining a friend
        '''
        # Store paramters
        self.data = data
        self.dist_thre = dist_thre
        self.ratio_thre = ratio_thre       
        # Get the list for all the unique frames
        self.frameList = np.unique(self.data[:, 0])
        # print('Frame list: ', self.frameList)
        # Get the list for all unique users
        self.userList = np.unique(self.data[:, 1])
           
    
    def get_frame_data(self, frameList):
        '''
        This is the function to get the data within the list of frames
        params:
            frameList: the list of the frames to be considered
        '''
        frameData = np.empty(shape=[0, 4])
        for frame in frameList:
            fData = self.data[self.data[:, 0]==frame, :]
            frameData = np.vstack((frameData, fData))
        return frameData
    
    
    def frame_DBscan(self, frameData, egoUserId, egoUserFl):
        '''
        This is the function to detect friend clusters based on each frame
        params:
            frameData: trajectories for the ego user and co-existing users
            egoUserId: the id for the given ego user
            egoUserFl: the list of frames the ego user appears 
        '''
        friendCandidate = np.empty(shape=[0, 3])
        for fl in egoUserFl:
            # Extract all the road users coordinates for the single frame
            frame = frameData[frameData[:, 0]==fl, :]
            # Using DBSCAN to cluster road users in the given frame
            clustering = DBSCAN(eps=self.dist_thre, min_samples=2).fit(frame[:, 2:4])
            labels = clustering.labels_
            cluster = np.concatenate((frame[:, 0:2], np.reshape(labels, (-1, 1))), axis=1)
            ego_clusterLabel = cluster[cluster[:, 1]==egoUserId, 2][0]
            
            # Only extract the cluster containing the ego user
            # need to check if the ego user is noise
            if ego_clusterLabel > -1:
                ego_cluster = cluster[cluster[:, 2]==ego_clusterLabel, :]
                friendCandidate = np.vstack((friendCandidate, ego_cluster))

            
        # Count the frequency for the friend Candidate staying in the same cluster as the ego user
        frequency = np.bincount(friendCandidate[:, 1].astype(int))
        userIds = np.nonzero(frequency)[0]
        
        userListLens = []
        for userId in userIds:
            userdata = self.data[self.data[:, 1]==userId, :]
            UserFl = np.unique(userdata[:, 0])
            wholeLength = len(np.unique(np.hstack((egoUserFl, UserFl))))        
            userListLens.append(wholeLength)
        userListLens = np.asarray(userListLens)
        
        DBscan_count = np.vstack((userIds,frequency[userIds], userListLens)).T
        friends = []
        for DB in DBscan_count:
            if (DB[1] / DB[2] > self.ratio_thre) and (DB[0] != egoUserId) and DB[2]>4:
                friends.append(DB[0])
        return friends
        
        
    
   
    
    
    
    
    
    
        
        
        
        
        
        
    
       
        

