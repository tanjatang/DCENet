# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 20:57:00 2020
@author: cheng,tang,liao
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import glob
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import sys
import time
from collision import check_collision
from datainfo import datainfo
from Dataloader import preprocess_data, loaddata
from evaluation import get_errors
from DCENet import dcenet
from mkdir import mak_dir
from plots import plot_pred
import writer
from ranking import gauss_rank


def main():
    desc = "Keras implementation of CVAE for trajectory prediction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--num_pred', type=int, default=25, help='This is the number of predictions for each agent')
    parser.add_argument('--obs_seq', type=int, default=8, help='Number of time steps observed')
    parser.add_argument('--enviro_pdim', type=int, default=[32, 32, 3],
                        help='The dimension of the environment after padding')
    parser.add_argument('--pred_seq', type=int, default=12, help='Number of time steps to be predicted')
    parser.add_argument('--dist_thre', type=float, default=1.0, help='The distance threhold for group detection')
    parser.add_argument('--ratio', type=float, default=0.95, help='The overlap ratio of coexisting for group detection')
    parser.add_argument('--n_hidden', type=int, default=512, help='This is the hidden size of the cvae')
    parser.add_argument('--z_dim', type=int, default=2, help='This is the size of the latent variable')
    parser.add_argument('--encoder_dim', type=int, default=16, help='This is the size of the encoder output dimension')
    parser.add_argument('--z_decoder_dim', type=int, default=64, help='This is the size of the decoder LSTM dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='The size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=192, help='Batch size')  # 192
    parser.add_argument('--o_drop', type=float, default=0.2, help='The dropout rate for occupancy')  # 0.2
    parser.add_argument('--s_drop', type=float, default=0.1, help='The dropout rate for trajectory sequence')  # 0.1
    parser.add_argument('--z_drop', type=float, default=0.15, help='The dropout rate for z input')  # 0.15
    parser.add_argument('--beta', type=float, default=0.65, help='Loss weight')  # 0.75#0.65
    parser.add_argument('--query_dim', type=int, default=4, help='The dimension of the query')
    parser.add_argument('--keyvalue_dim', type=int, default=4, help='The dimension for key and value')
    parser.add_argument('--train_mode', type=bool, default=True, help='This is the training mode')
    parser.add_argument('--train_set', type=str, choices=['Train'], default='Train',
                        help='This is the directories for the training data')
    parser.add_argument('--challenge_set', type=str, choices=['Test'], default='Test',
                        help='This is the directories for the challenge data')  # it is the online test set
    parser.add_argument('--split', type=float, default=0.8, help='the split rate for training and validation')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')  # 3-5
    parser.add_argument('--aug_num', type=int, default=4, help='Number of augmentations')
    parser.add_argument('--epochs', type=int, default=100, help='Number of batches')
    parser.add_argument('--patience', type=int, default=5,
                        help='Maximum mumber of continuous epochs without converging')
    parser.add_argument('--x_encoder_dim', type=int, default=64, help='Mapping dimension of x_encoder')
    parser.add_argument('--y_encoder_dim', type=int, default=64, help='Mapping dimension of y_encoder')
    parser.add_argument('--x_encoder_layers', type=int, default=3, help='Number of transformer block layers for x_encoder')
    parser.add_argument('--y_encoder_layers', type=int, default=3, help='Number of transformer block layer for y_encoder')
    parser.add_argument('--x_encoder_head', type=int, default=8, help='Head number of x_encoder')
    parser.add_argument('--y_encoder_head', type=int, default=8, help='Head number of y_encoder')
    parser.add_argument('--occu_encoder_x_dim', type=int, default=32, help='Mapping dimension of the x occupancy encoder')
    parser.add_argument('--occu_encoder_y_dim', type=int, default=32, help='Mapping dimension of the y occupancy encoder')
    parser.add_argument('--occu_encoder_x_layers', type=int, default=3,
                        help='Number of transformer block layers for x_encoder')
    parser.add_argument('--occu_encoder_y_layers', type=int, default=3,
                        help='Number of transformer block layer for y_encoder')
    parser.add_argument('--occu_encoder_x_head', type=int, default=2, help='Head number of x_encoder')
    parser.add_argument('--occu_encoder_y_head', type=int, default=2, help='Head number of y_encoder')
    parser.add_argument('--preprocess_data', type=bool, default=False, help='Process and merge dataset')

    args = parser.parse_args(sys.argv[1:])


    # Make all the necessary folders
    mak_dir()

    # # specify the directory for training and challenge data
    # ToDo chenge this to make compatible with linus
    train_paths = sorted(glob.glob("../WORLD H-H TRAJ/%s/**/*.txt" % (args.train_set)))
    # train_paths= sorted(glob.glob("../trajectories/%s/**/*.txt"%(args.train_set)))
    # # NOTE, here the challenge set is the "ONLINE" test set
    # # This is different from the "OFFLINE" test set
    # ToDo chenge this to make compatible with linus
    challenge_paths = sorted(glob.glob("../WORLD H-H TRAJ/%s/**/*.txt" % (args.challenge_set)))
    # challenge_paths = sorted(glob.glob("../trajectories/%s/**/*.txt"%(args.challenge_set)))

    # Process the data
    for path in train_paths:
        # dataname = path.split('\\')[-1].split('.')[0]
        # ToDo chenge this to make compatible with linus
        dataname = os.path.splitext(os.path.basename(path))[0]
        # ToDo chenge this to make compatible with linus
        # if not os.path.exists("/phys/ssd/tangxueq/processed_data/train/%s.npz"%dataname):
        if not os.path.exists("../processed_data/train/%s.npz" % dataname):
            # preprocess_data(path, args.obs_seq+args.pred_seq-1, args.enviro_pdim, "train")
            preprocess_data(seq_length=args.obs_seq + args.pred_seq - 1,
                            size=args.enviro_pdim,
                            dirname="train",
                            path=path,
                            aug_num=args.aug_num,
                            save=True)

    for path in challenge_paths:
        # dataname = path.split('\\')[-1].split('.')[0]
        dataname = os.path.splitext(os.path.basename(path))[0]
        # ToDo chenge this to make compatible with linus
        if not os.path.exists("/phys/ssd/slurmstorage/tangxueq/processed_data/challenge/%s.npz" % dataname):
            # preprocess_data(path, args.obs_seq-1, args.enviro_pdim, "challenge")
            preprocess_data(seq_length=args.obs_seq - 1,
                            size=args.enviro_pdim,
                            dirname="challenge",
                            path=path,
                            save=True)

    # Check the daatinfo for dataset partition
    Datalist = datainfo()
    # Datalist = datainfo(4)

    # Define the callback and early stop
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # ToDo chenge this to make compatible with linus
    filepath = "/phys/ssd/slurmstorage/tangxueq/models/DCENet_%0.f_%s.hdf5" % (args.epochs, timestr)
    ## Eraly stop
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=100, mode='auto')
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [earlystop, checkpoint]

    # # Instantiate the model
    DCENet = dcenet(args)
    # Contruct the cave model
    train = DCENet.training()
    train.summary()

    x_encoder = DCENet.X_encoder()
    decoder = DCENet.Decoder()
    # sys.exit()

    # Start training phase
    if args.train_mode:
        if args.preprocess_data:
            traindata_list = Datalist.train_data
        else:
            traindata_list = Datalist.train_merged

        print("train data list", traindata_list)

        # # NOTE: this is the "OFFLINE" test set. This is only used to plot if the prediction is feasible
        # # This test set has nothing to do with the challenge data set ("ONLINE" test set)
        testdata_list = Datalist.train_biwi
        print("test data list", testdata_list)

        # Get the data fro training and validation
        np.random.seed(10)
        offsets, traj_data, occupancy = loaddata(traindata_list, args, datatype="train")
        train_val_split = np.random.rand(len(offsets)) < args.split

        train_x = offsets[train_val_split, :args.obs_seq - 1, 4:6]
        train_occu = occupancy[train_val_split, :args.obs_seq - 1, ..., :args.enviro_pdim[-1]]
        train_y = offsets[train_val_split, args.obs_seq - 1:, 4:6]
        train_y_occu = occupancy[train_val_split, args.obs_seq - 1:, ..., :args.enviro_pdim[-1]]

        val_x = offsets[~train_val_split, :args.obs_seq - 1, 4:6]
        val_occu = occupancy[~train_val_split, :args.obs_seq - 1, ..., :args.enviro_pdim[-1]]
        val_y = offsets[~train_val_split, args.obs_seq - 1:, 4:6]
        val_y_occu = occupancy[~train_val_split, args.obs_seq - 1:, ..., :args.enviro_pdim[-1]]

        print("%.0f trajectories for training\n %.0f trajectories for valiadation" %
              (train_x.shape[0], val_x.shape[0]))

        test_offsets, test_trajs, test_occupancy = loaddata(testdata_list, args, datatype="test")
        test_x = test_offsets[:, :args.obs_seq - 1, 4:6]
        test_occu = test_occupancy[:, :args.obs_seq - 1, ..., :args.enviro_pdim[-1]]
        last_obs_test = test_offsets[:, args.obs_seq - 2, 2:4]
        y_truth = test_offsets[:, args.obs_seq - 1:, :4]
        xy_truth = test_offsets[:, :, :4]
        print("%.0f trajectories for testing" % (test_x.shape[0]))

        print("Start training the model...")
        # Retrain from last time
        # train.load_weights("/home/tangxueq/MA_tang/trajnet_challenge/best.hdf5")
        train.fit(x=[train_occu, train_x, train_y_occu, train_y],
                  y=train_y,
                  shuffle=True,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  verbose=1,
                  callbacks=callbacks_list,
                  validation_data=([val_occu, val_x, val_y_occu, val_y], val_y))
        train.load_weights(filepath)

        # Start inference phase
        # Retrieve the x_encoder and the decoder
        x_encoder = DCENet.X_encoder()
        decoder = DCENet.Decoder()
        x_encoder.summary()
        decoder.summary()

        # get the x_encoded_dense as latent feature for prediction
        x_latent = x_encoder.predict([test_occu, test_x], batch_size=args.batch_size)
        # x_latent = x_encoder.predict(test_x, batch_size=args.batch_size)

        # Using x_latent and z as input of the decoder for generating future trajectories
        print("Start predicting")
        predictions = []
        for i, x_ in enumerate(x_latent):
            last_pos = last_obs_test[i]
            x_ = np.reshape(x_, [1, -1])
            for i in range(args.num_pred):
                # sampling z from a normal distribution
                z_sample = np.random.rand(1, args.z_dim)
                y_p = decoder.predict(np.column_stack([z_sample, x_]))
                y_p_ = np.concatenate(([last_pos], np.squeeze(y_p)), axis=0)
                y_p_sum = np.cumsum(y_p_, axis=0)
                predictions.append(y_p_sum[1:, :])
        predictions = np.reshape(predictions, [-1, args.num_pred, args.pred_seq, 2])

        print('Predicting done!')
        print(predictions.shape)
        plot_pred(xy_truth, predictions)
        # Get the errors for ADE, DEF, Hausdorff distance, speed deviation, heading error
        print("\nEvaluation results @top%.0f" % args.num_pred)
        errors = get_errors(y_truth, predictions)
        check_collision(y_truth)

        ##
        ## Get the first time prediction by g
        ranked_prediction = []
        for prediction in predictions:
            ranks = gauss_rank(prediction)
            ranked_prediction.append(prediction[np.argmax(ranks)])
        ranked_prediction = np.reshape(ranked_prediction, [-1, 1, args.pred_seq, 2])
        print("\nEvaluation results for most-likely predictions")
        # ranked_errors = get_errors(y_truth, ranked_prediction)


    else:
        print('Run pretrained model')
        train.load_weights("../best.hdf5")

    challenge_list = Datalist.challenge_data
    for challenge_dataname in challenge_list:
        print(challenge_dataname, "\n")
        challenge_offsets, challenge_trajs, challenge_occupancy = loaddata([challenge_dataname], args,
                                                                           datatype="challenge")
        print(challenge_offsets.shape, challenge_trajs.shape, challenge_occupancy.shape)

        challenge_x = challenge_offsets[:, :args.obs_seq - 1, 4:6]
        challenge_occu = challenge_occupancy[:, :args.obs_seq - 1, ..., :args.enviro_pdim[-1]]
        last_obs_challenge = challenge_trajs[:, args.obs_seq - 1, 2:4]
        print("%.0f trajectories for challenge" % (challenge_x.shape[0]))

        # Start inference phase
        # Retrieve the x_encoder and the decoder
        x_encoder = DCENet.X_encoder()
        decoder = DCENet.Decoder()


        # get the x_encoded_dense as latent feature for prediction
        x_latent = x_encoder.predict([challenge_occu, challenge_x], batch_size=args.batch_size)

        # Using x_latent and z as input of the decoder for generating future trajectories
        print("Start predicting the challenge data")
        challenge_predictions = []
        for i, x_ in enumerate(x_latent):
            last_pos = last_obs_challenge[i]
            x_ = np.reshape(x_, [1, -1])
            for i in range(args.num_pred):
                # sampling z from a normal distribution
                z_sample = np.random.rand(1, args.z_dim)
                y_p = decoder.predict(np.column_stack([z_sample, x_]))
                y_p_ = np.concatenate(([last_pos], np.squeeze(y_p)), axis=0)
                y_p_sum = np.cumsum(y_p_, axis=0)
                challenge_predictions.append(y_p_sum[1:, :])
        challenge_predictions = np.reshape(challenge_predictions, [-1, args.num_pred, args.pred_seq, 2])

        print('Predicting done!')
        print(challenge_predictions.shape)

        ##
        ## Get the first time prediction
        challenge_ranked_prediction = []
        for prediction in challenge_predictions:
            ranks = gauss_rank(prediction)
            challenge_ranked_prediction.append(prediction[np.argmax(ranks)])
        challenge_ranked_prediction = np.reshape(challenge_ranked_prediction, [-1, 1, args.pred_seq, 2])

        challenge_pred_traj = writer.get_index(challenge_trajs, challenge_predictions)
        print("Collision in ranked prediction")
        check_collision(np.squeeze(challenge_pred_traj))
        writer.write_pred_txt(challenge_trajs, challenge_predictions, challenge_dataname, "prediction")

if __name__ == "__main__":
    main()
