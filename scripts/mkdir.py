# -*- coding: utf-8 -*-
import os

def mak_dir():
    # Make all the folders to save the intermediate results
    model_dir = "../models"
    processed_train = "../processed_data/train"
    processed_challenge = "../processed_data/challenge"
    # Save the DCENet model's prediction
    prediction = "../prediction"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('%s created'%model_dir)
    if not os.path.exists(processed_train):
        os.makedirs(processed_train)
        print('%s created'%processed_train)
    if not os.path.exists(processed_challenge):
        os.makedirs(processed_challenge)
        print('%s created'%processed_challenge)
    if not os.path.exists(prediction):
        os.makedirs(prediction)
        print('%s created'%prediction)

if __name__ == "__main__":
    mak_dir()
