# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal


def gauss_rank(pred_trajs, addnoise=False):
    '''
    pred_trajs: numberofpredictions*length*[x, y]
    '''
    # Swap time axis to the first
    pred_trajs_t = np.swapaxes(pred_trajs, 1, 0)
    rank = np.zeros((0, pred_trajs.shape[0]))
    for pred_poss in pred_trajs_t:
 
        # pred_poss is the sampled positions at each time step
        # pred_poss will be used to fit a bivariable gaussian distribution 
        if addnoise == True:
            pred_poss = pred_poss + np.random.normal(0, 1, pred_poss.shape)
        mu = np.mean(pred_poss, axis=0)
        covariance = np.cov(pred_poss.T)
        pos_pdf = multivariate_normal.pdf(pred_poss, mean=mu, cov=covariance)
        rank = np.vstack((rank, pos_pdf))
    rank = np.mean(np.log(rank), axis=0)
    return rank
