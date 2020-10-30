# -*- coding: utf-8 -*-


import glob
import os

class datainfo():
    
    def __init__(self):
        
        all_traindata_dirs = sorted(glob.glob(os.path.join("../processed_data/train", "*.npz")))
        all_challengedata_dirs = sorted(glob.glob(os.path.join("../processed_data/challenge", "*.npz")))
        train_data = []
        challenge_data = []
        for train_dir in all_traindata_dirs:
            train_dataname = os.path.splitext(os.path.basename(train_dir))[0]
            if train_dataname not in train_data:
                train_data.append(train_dataname)
                
        for challenge_dir in all_challengedata_dirs:
            challenge_dataname = os.path.splitext(os.path.basename(challenge_dir))[0]
            if challenge_dataname not in challenge_data:
                challenge_data.append(challenge_dataname)
                               
        self.train_data = train_data        
        self.challenge_data = challenge_data
        
        self.train_biwi = ["biwi_hotel"]
        
        self.train_crowds = ["crowds_zara03",
                             "crowds_zara02",                              
                             "arxiepiskopi1",
                             "students001", 
                             "students003"]
        
        self.train_sdd_roundabout = ["deathCircle_0", 
                                     "deathCircle_1", 
                                     "deathCircle_3", 
                                     "deathCircle_4",
                                     "gates_1",
                                     "gates_3"]
        
        self.train_sdd = ["bookstore_0",
                          "bookstore_1",
                          "bookstore_2",
                          "bookstore_3",
                          "coupa_3",
                          "deathCircle_2",
                          "gates_0",
                          "gates_4",
                          "gates_5",
                          "gates_6",
                          "gates_7",
                          "gates_8",
                          "hyang_4",
                          "hyang_5",
                          "hyang_6",
                          "hyang_7",
                          "hyang_9",
                          "nexus_0",
                          "nexus_1",
                          "nexus_2",
                          "nexus_3",
                          "nexus_4",
                          "nexus_7",
                          "nexus_8",
                          "nexus_9"]
        self.train_merged = ["train_merged"]
        
        
        self.challenge_biwi = ["biwi_eth"] 
        
        self.challenge_crowds = ["crowds_zara01",
                                 "uni_examples"]
        
        self.challenge_sdd_roundabout = ["gates_2"]
        
        self.challenge_sdd = ["coupa_0",
                              "coupa_1",
                              "hyang_0",
                              "hyang_1",
                              "hyang_3",
                              "hyang_8",
                              "little_0",
                              "little_1",
                              "little_2",
                              "little_3",
                              "nexus_5",
                              "nexus_6",
                              "quad_0",
                              "quad_1",
                              "quad_2",
                              "quad_3"]

        
        self.challenge_test= ["crowds_zara01"]
        
        self.challenge_mix = ["biwi_eth", 
                              "crowds_zara01",
                              "uni_examples"]
        
        
        
        
        
        
    
        
        
                
            
                
            
                
            
            