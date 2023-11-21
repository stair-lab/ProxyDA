"""
Adaptation pipeline of the deep kernel implementation.
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT


import torch
from torch import nn
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from .deep_model import DeepFullGraph, DeepPartialGraph
from .trainer import FullDeepKernelTrainer, PartialDeepKernelTrainer
from .torch_reg_utils import linear_reg_pred

logger = logging.getLogger()


class DeepKernelMethod:
    """ base estimator for the adaptation
    """
    def __init__(self, config_files, gpu_flg):
        """
        Args:
            data_config: .json file
            
        """
        with open(config_files) as f:
            config = json.load(f)
            data_configs = config["data"]
            train_params = config["model"]
        self.fitted = False
        self.data_configs = data_configs
        self.train_params = train_params
        self.gpu_flg = gpu_flg


    def fit(self, source_traindata, target_traindata=None, split=False, verbose=2):
        """fit the graph with deep kernel features.
        Args:
            source_traindata: dfaDataSetTorch
            target_traindata: dfaDataSetTorch or None
            split: split training data or not, Boolean
            verbose: int
        """

        model = self.trianer.train(source_traindata, target_traindata, split, verbose)
        self.model = model
        
        self.w_dim = self.model.w_dim
        self.fitted = True

    def score(self, testY, predictY):
        """l2 error of the prediciton.
        Args:
            testY: true Y, torch.Tensor
            predictY: predict Y, torch.Tensor
        """
        ## Fix shape
        if testY.shape > predictY.shape:
            assert testY.ndim == predictY.ndim + 1 and testY.shape[:-1] == predictY.shape, "unresolveable shape mismatch betweenn testY and predictY"
            predictY = predictY.reshape(testY.shape)
        elif testY.shape < predictY.shape:
            assert testY.ndim + 1 == predictY.ndim and testY.shape == predictY.shape[:-1], "unresolveable shape mismatch betweenn testY and predictY"
            testY = testY.reshape(predictY.shape)
        l2_error =  torch.sum((testY-predictY)**2)/predictY.shape[0]
        return l2_error
    

    def predict(self):
        """Fits the model to the training data."""
        raise NotImplementedError("Implemented in child class.")

    def evaluation(self):
        """Fits the model to the training data."""
        raise NotImplementedError("Implemented in child class.")



class deep_full_adaptation(DeepKernelMethod):
    def __init__(self, config_files, gpu_flg):
        
        DeepKernelMethod.__init__(self, config_files, gpu_flg)
        
        self.trianer = FullDeepKernelTrainer(self.data_configs, self.train_params, gpu_flg)
        #self.target_trianer = FullDeepKernelTrainer(self.data_configs, self.train_params, gpu_flg)

    def predict(self, testX, cme_wc_x_domain):
        """ predict the ourcome. 
        Args:
            testX: covarites, torch.Tensor
            cme_wc_x_domain: specify domain, 'original' or 'adapt'
        """

        assert(self.fitted == True)
        if cme_wc_x_domain == 'original':
            test_x1_feature = self.model.x1_source_feature_net(testX)
            test_x1_feature = DeepFullGraph.augment_single_feature(test_x1_feature, self.model.add_cme_intercept)
            ptest_wc1_feature = linear_reg_pred(test_x1_feature, self.model.source_coef_wc_x)
        
        elif cme_wc_x_domain == 'adapt':
            test_x1_feature = self.model.x1_target_feature_net(testX)
            test_x1_feature = DeepFullGraph.augment_single_feature(test_x1_feature, self.model.add_cme_intercept)
            ptest_wc1_feature = linear_reg_pred(test_x1_feature, self.model.target_coef_wc_x)    

        """
        ptest_w1_feature  = ptest_wc1_feature[:, 0:self.w_dim]
        ptest_c1_feature  = ptest_wc1_feature[:, self.w_dim::]
        
        feature = DeepFullGraph.augment_wc_feature(ptest_w1_feature,
                                                    ptest_c1_feature,
                                                    self.model.add_h0_intercept)
        pred = linear_reg_pred(feature, self.model.coef_h0)
        """
        pred = linear_reg_pred(ptest_wc1_feature, self.model.coef_h0)

        
        return pred



class deep_partial_adaptation(DeepKernelMethod):
    def __init__(self, config_files, gpu_flg):
        
        DeepKernelMethod.__init__(self, config_files, gpu_flg)
        
        self.trianer = PartialDeepKernelTrainer(self.data_configs, self.train_params, gpu_flg)
        #self.target_trianer = FullDeepKernelTrainer(self.data_configs, self.train_params, gpu_flg)

    def predict(self, testX, cme_w_x_domain):
        """ predict the ourcome. 
        Args:
            testX: covarites, torch.Tensor
            cme_wc_x_domain: specify domain, 'original' or 'adapt'
        """

        assert(self.fitted == True)
        if cme_w_x_domain == 'original':
            test_x1_feature = self.model.x1_source_feature_net(testX)
            test_x1_feature = DeepFullGraph.augment_single_feature(test_x1_feature, self.model.add_cme_intercept)
            ptest_w1_feature = linear_reg_pred(test_x1_feature, self.model.source_coef_w_x)

        elif cme_w_x_domain == 'adapt':
            test_x1_feature = self.model.x1_target_feature_net(testX)
            test_x1_feature = DeepFullGraph.augment_single_feature(test_x1_feature, self.model.add_cme_intercept)
            ptest_w1_feature = linear_reg_pred(test_x1_feature, self.model.target_coef_w_x)    

        test_x4_feature = self.model.x4_feature_net(testX)

        feature = DeepPartialGraph.augment_wx_feature(ptest_w1_feature,
                                                    test_x4_feature,
                                                   self.model.add_m0_intercept)


        m0_c_feature = linear_reg_pred(feature, self.model.coef_m0)

        
        feature = DeepFullGraph.augment_wc_feature(ptest_w1_feature,
                                                   m0_c_feature,
                                                   self.model.add_h0_intercept)
        pred = linear_reg_pred(feature, self.model.coef_h0)
       
        return pred
        
 