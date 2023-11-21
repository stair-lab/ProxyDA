"""
Trainer of the deep kernel method.
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from typing import Dict, Any, Optional
import torch
from torch import nn
import logging
from pathlib import Path

import numpy as np
from .torch_reg_utils import linear_reg_loss
from .deep_model import DeepFullGraph, DeepPartialGraph
from .nn_structure import build_extractor
from data.data_class import  split_train_data, dfaDataSetTorch

import matplotlib.pyplot as plt
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class FullDeepKernelTrainer:
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False):
        
        
        self.track_loss = {'h0':[], 'w_xc':[], 'wc_x.x':[], 'wc_x.x_target':[], 'wc_x.wc':[]}
        
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params

        self.lam_set: dict = train_params["lam_set"]
        self.cme_iter: int = train_params["cme_iter"]
        self.h0_iter: int = train_params["h0_iter"]

        self.n_epoch: int = train_params["n_epoch"]
        self.add_cme_intercept = True
        self.add_h0_intercept = True
        self.w_weight_decay = train_params["w_weight_decay"]
        self.x_weight_decay = train_params["x_weight_decay"]
        self.c_weight_decay = train_params["c_weight_decay"]
        self.w_lr = train_params["w_lr"]
        self.x_lr = train_params["x_lr"]
        self.c_lr = train_params["c_lr"]


        # build networks
        networks = build_extractor(data_configs["name"])
        self.x1_source_feature_net: nn.Module = networks[0]
        self.x1_target_feature_net:  Optional[nn.Module] = networks[1]

        self.x2_feature_net: nn.Module = networks[2]
        
        #self.w1_feature_net: nn.Module = networks[2]
        self.w2_feature_net: nn.Module = networks[3]

        #self.c1_feature_net: nn.Module = networks[4]
        self.c2_feature_net: nn.Module = networks[4]
        self.c3_feature_net: nn.Module = networks[5]


        if self.gpu_flg:
            self.x1_source_feature_net.to("cuda")
            
            if self.x1_target_feature_net is not None:
                self.x1_target_feature_net.to("cuda:0")
            self.x2_feature_net.to("cuda")
    
            #self.w1_feature_net.to("cuda:0")
            self.w2_feature_net.to("cuda")
    
            #self.c1_feature_net.to("cuda:0")
            self.c2_feature_net.to("cuda")
            self.c3_feature_net.to("cuda")
   
        self.x1_source_opt = torch.optim.Adam(self.x1_source_feature_net.parameters(),
                                        weight_decay=self.x_weight_decay,
                                        lr=self.x_lr)
        if self.x1_target_feature_net:
            self.x1_target_opt = torch.optim.Adam(self.x1_target_feature_net.parameters(),
                                            weight_decay=self.x_weight_decay,
                                            lr=self.x_lr)

        self.x2_opt = torch.optim.Adam(self.x2_feature_net.parameters(),
                                        weight_decay=self.x_weight_decay,
                                        lr=self.x_lr)

        #self.w1_opt = torch.optim.Adam(self.w1_feature_net.parameters(),
        #                                weight_decay=self.w_weight_decay)
        
        self.w2_opt = torch.optim.Adam(self.w2_feature_net.parameters(),
                                        weight_decay=self.w_weight_decay,
                                        lr=self.w_lr)


        #self.c1_opt = torch.optim.Adam(self.c1_feature_net.parameters(),
        #                                weight_decay=self.c_weight_decay)
        
        self.c2_opt = torch.optim.Adam(self.c2_feature_net.parameters(),
                                        weight_decay=self.c_weight_decay,
                                        lr=self.c_lr)

        self.c3_opt = torch.optim.Adam(self.c3_feature_net.parameters(),
                                        weight_decay=self.c_weight_decay,
                                        lr=self.c_lr)
        
    def train(self, source_traindata, target_traindata, split, verbose, PLOT=True):
        """
        Args:
            source_traindata: dfaDataSetTorch
            target_traindata: dfaDataSetTorch
            split: Boolean
            verbose: int
            PLOT: Boolean
        """
        if split:
            train_set = split_train_data(source_traindata, 3)
            source_traindata1 = train_set[0]
            source_traindata2 = train_set[1] 
            source_traindata3 = train_set[2]
            if self.gpu_flg:
                source_traindata1 = source_traindata1.to_gpu()
                source_traindata2 = source_traindata2.to_gpu()
                source_traindata3 = source_traindata3.to_gpu()
                if target_traindata is not None:
                    target_traindata = target_traindata.to_gpu()

            for t in range(self.n_epoch):
                self.cme_w_xc_feature_update(source_traindata2, verbose)
                self.h0_feature_update(source_traindata2, source_traindata3, verbose)
                self.cme_wc_x_feature_update(source_traindata1, target_traindata, verbose)

                #self.cme_wc_x_outcome_update(train_data1, train_data2, train_data3, verbose)
                if verbose >= 1:
                    logger.info(f"Epoch {t} ended")    

    
            

        else:
            if self.gpu_flg:
                source_traindata = source_traindata.to_gpu()
            
            for t in range(self.n_epoch):
                self.cme_w_xc_feature_update(source_traindata, verbose)
                self.h0_feature_update(source_traindata, source_traindata, verbose)
                self.cme_wc_x_feature_update(source_traindata, target_traindata, verbose)
                #self.cme_wc_x_outcome_update(train_data, train_data, train_data3, verbose)
                if verbose >= 1:
                    logger.info(f"Epoch {t} ended")
        
        if PLOT:
            plt.figure()
            plt.plot(self.track_loss['h0'], label='h0')
            plt.xlabel('iteration')
            plt.ylabel('h0 loss')
            plt.yscale('log')
            plt.savefig('h0_loss.png', bbox_inches='tight')   

            plt.figure()   
            plt.plot(self.track_loss['w_xc'], label='w_xc')
            plt.xlabel('iteration')
            plt.ylabel('w_xc loss')
            plt.yscale('log')
            plt.savefig('cme_w_xc_loss.png', bbox_inches='tight')   

        
            plt.figure()   
            plt.plot(self.track_loss['wc_x.x'], label='wc_x.x')
            plt.xlabel('iteration')
            plt.ylabel('wc_x.x loss')
            plt.yscale('log')
            plt.savefig('cme_wc_x_x_source.png', bbox_inches='tight') 

            plt.figure()   
            plt.plot(self.track_loss['wc_x.x_target'], label='wc_x.x')
            plt.xlabel('iteration')
            plt.ylabel('wc_x.x_target loss')
            plt.yscale('log')
            plt.savefig('cme_wc_x_x_target.png', bbox_inches='tight') 
        
            #plt.figure()   
            #plt.plot(self.track_loss['wc_x.wc'], label='wc_x.wc')
            #plt.xlabel('iteration')
            #plt.ylabel('loss')
            #plt.yscale('log')
            #plt.savefig('cme_wx_x_wc_loss.png', bbox_inches='tight')   
            
        #need to be completed
        dfa = DeepFullGraph(self.x1_source_feature_net, 
                    self.x1_target_feature_net, 
                    self.x2_feature_net, 
                    self.w2_feature_net, 
                    self.c2_feature_net, 
                    self.c3_feature_net,
                    self.add_cme_intercept, 
                    self.add_h0_intercept)
        
        #fit thr trained model 
        if split: 
            dfa.fit_t(source_traindata1, source_traindata2, source_traindata3, target_traindata, self.lam_set['cme'], self.lam_set['h0'])
        else:
            dfa.fit_t(source_traindata, source_traindata, source_traindata, target_traindata, self.lam_set['cme'], self.lam_set['h0'])
        return dfa


    def cme_wc_x_feature_update(self, source_traindata, target_traindata, verbose=-1):
        """
        Estimate cme(wc|x), update x feature kernel
        """
        self.x1_source_feature_net.train(True)

        if self.x1_target_feature_net:
            self.x1_target_feature_net.train(True)
        
        #self.w1_feature_net.train(False)
        #self.c1_feature_net.train(False)

        self.x2_feature_net.train(False)
        self.w2_feature_net.train(False)
        self.c2_feature_net.train(False)

        self.c3_feature_net.train(False)

        with torch.no_grad():
            w2_feature = self.w2_feature_net(source_traindata.W)
            c3_feature = self.c3_feature_net(source_traindata.C)

            """            
            if len(w2_feature.shape) == 1:
                w2_feature = w2_feature[:, None]
            if len(c3_feature.shape) == 1:
                c3_feature = c3_feature[:, None]
            
            w2c3_feature = torch.cat((w2_feature, c3_feature), 1)
            """
            w2c3_feature = DeepFullGraph.augment_wc_feature(w2_feature, c3_feature, self.add_h0_intercept)
            if target_traindata is not None:
                
                w2_target_feature = self.w2_feature_net(target_traindata.W)
                c3_target_feature = self.c3_feature_net(target_traindata.C)
                """
                if len(w2_target_feature.shape) == 1:
                    w2_target_feature = w2_target_feature[:, None]
                if len(c3_target_feature.shape) == 1:
                    c3_target_feature = c3_target_feature[:, None]

                w2c3_target_feature = torch.cat((w2_target_feature, c3_target_feature), 1)    
                """
                w2c3_target_feature = DeepFullGraph.augment_wc_feature(w2_target_feature, c3_target_feature, self.add_h0_intercept)
        
        for i in range(self.cme_iter):
            self.x1_source_opt.zero_grad()
            if self.x1_target_feature_net and target_traindata is not None:
                self.x1_target_opt.zero_grad()
                
                x1_target_feature = self.x1_target_feature_net(target_traindata.X)
                x1_target_feature = DeepFullGraph.augment_single_feature(x1_target_feature, self.add_cme_intercept)


            x1_feature = self.x1_source_feature_net(source_traindata.X)
            x1_feature = DeepFullGraph.augment_single_feature(x1_feature, self.add_cme_intercept)



            loss =  linear_reg_loss(w2c3_feature, x1_feature, self.lam_set['cme'])

            self.track_loss['wc_x.x'].append(loss.item())
            
            if self.x1_target_feature_net and target_traindata is not None:
                l2 = linear_reg_loss(w2c3_target_feature, x1_target_feature, self.lam_set['cme'])
                loss += l2
                self.track_loss['wc_x.x_target'].append(l2.item())

            loss.backward()
            
            
            if verbose >= 2:
                logger.info(f"cme_wc_x x learning: {loss.item()}")

            self.x1_source_opt.step()
            if self.x1_target_feature_net:
                self.x1_target_opt.step()



    def cme_w_xc_feature_update(self, train_data, verbose):
        """
        Estimate cme(w|c,x), update c,x feature kernel
        """
        self.x1_source_feature_net.train(False)
        if self.x1_target_feature_net:
            self.x1_target_feature_net.train(False)
        #self.w1_feature_net.train(False)
        #self.c1_feature_net.train(False)

        self.x2_feature_net.train(True)
        self.w2_feature_net.train(False)
        self.c2_feature_net.train(True)

        self.c3_feature_net.train(False)

        with torch.no_grad():
            w2_feature = self.w2_feature_net(train_data.W)
        
        for _ in range(self.cme_iter):
            self.x2_opt.zero_grad()
            self.c2_opt.zero_grad()
            x2_feature = self.x2_feature_net(train_data.X)
            c2_feature = self.c2_feature_net(train_data.C)
            features = DeepFullGraph.augment_xc_feature(x2_feature, c2_feature, self.add_cme_intercept)
            
            loss = linear_reg_loss(w2_feature, features, self.lam_set['cme'])
            loss.backward()
            self.track_loss['w_xc'].append(loss.item())
            if verbose >= 2:
                logger.info(f"cme_w_xc xc learning: {loss.item()}")
            self.x2_opt.step()
            self.c2_opt.step()



    def h0_feature_update(self, train_data1, train_data2, verbose):
        """
        Estimate h0, cme(w|c,x), update w feature kernel, update h0
        """
        self.x1_source_feature_net.train(False)
        if self.x1_target_feature_net:
            self.x1_target_feature_net.train(False)
        #self.w1_feature_net.train(False)
        #self.c1_feature_net.train(False)

        self.x2_feature_net.train(False)
        self.w2_feature_net.train(True)
        self.c2_feature_net.train(False)

        self.c3_feature_net.train(True)
        
        with torch.no_grad():
            c2_feature1 = self.c2_feature_net(train_data1.C)
            x2_feature1 = self.x2_feature_net(train_data1.X)

            c2_feature2 = self.c2_feature_net(train_data2.C)
            x2_feature2 = self.x2_feature_net(train_data2.X)
        
        for _ in range(self.h0_iter):
            self.c3_opt.zero_grad()
            self.w2_opt.zero_grad()
            w2_feature1 = self.w2_feature_net(train_data1.W)
            c3_feature2 = self.c3_feature_net(train_data2.C)

            res = DeepFullGraph.fit_h0(x2_feature1, x2_feature2,
                                           c2_feature1, c2_feature2,
                                           c3_feature2, w2_feature1, #None, #not passing w2_feasure2 for training
                                           train_data2.Y, self.lam_set['cme'], self.lam_set['h0'],
                                           self.add_cme_intercept,
                                           self.add_h0_intercept)
            loss = res['loss']
            loss.backward()# update Model2 parameters
            self.coef_h0 = res['alpha']
            self.track_loss['h0'].append(loss.item())
            if verbose >= 2:
                logger.info(f"h0 learning: {loss.item()}")
            self.c3_opt.step()
            self.w2_opt.step()
        #return res['alpha']




class PartialDeepKernelTrainer(FullDeepKernelTrainer):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False):
        
        
        self.track_loss = {'h0':[], 'w_xc':[], 'w_x.x':[], 'w_x.x_target':[], 'm0':[]}
        
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params

        self.lam_set: dict = train_params["lam_set"]
        self.cme_iter: int = train_params["cme_iter"]
        self.h0_iter: int = train_params["h0_iter"]
        self.m0_iter: int = train_params["m0_iter"]
        self.add_cme_intercept = True
        self.add_h0_intercept = True
        self.add_m0_intercept = True
        self.add_h0_intercept = True
        self.n_epoch: int = train_params["n_epoch"]

        self.w_weight_decay = train_params["w_weight_decay"]
        self.x_weight_decay = train_params["x_weight_decay"]
        self.c_weight_decay = train_params["c_weight_decay"]
        self.w_lr = train_params["w_lr"]
        self.x_lr = train_params["x_lr"]
        self.c_lr = train_params["c_lr"]


        # build networks
        networks = build_extractor(data_configs["name"])
        self.x1_source_feature_net: nn.Module = networks[0]
        self.x1_target_feature_net:  Optional[nn.Module] = networks[1]

        self.x2_feature_net: nn.Module = networks[2]
        
        self.x4_feature_net: nn.Module = networks[3]
        
        self.w2_feature_net: nn.Module = networks[4]

        self.c2_feature_net: nn.Module = networks[5]
        self.c3_feature_net: nn.Module = networks[6]


        if self.gpu_flg:
            self.x1_source_feature_net.to("cuda:0")
            
            if self.x1_target_feature_net is not None:
                self.x1_target_feature_net.to("cuda:0")
            
            self.x2_feature_net.to("cuda:0")
            self.x4_feature_net.to("cuda:0")

            self.w2_feature_net.to("cuda:0")
    
            self.c2_feature_net.to("cuda:0")
            self.c3_feature_net.to("cuda:0")
   
        self.x1_source_opt = torch.optim.Adam(self.x1_source_feature_net.parameters(),
                                        weight_decay=self.x_weight_decay,
                                        lr=self.x_lr)
        if self.x1_target_feature_net:
            self.x1_target_opt = torch.optim.Adam(self.x1_target_feature_net.parameters(),
                                            weight_decay=self.x_weight_decay,
                                            lr=self.x_lr)

        self.x2_opt = torch.optim.Adam(self.x2_feature_net.parameters(),
                                        weight_decay=self.x_weight_decay,
                                        lr=self.x_lr)

        self.x4_opt = torch.optim.Adam(self.x4_feature_net.parameters(),
                                        weight_decay=self.x_weight_decay,
                                        lr=self.x_lr)

        self.w2_opt = torch.optim.Adam(self.w2_feature_net.parameters(),
                                        weight_decay=self.w_weight_decay,
                                        lr=self.w_lr)

        self.c2_opt = torch.optim.Adam(self.c2_feature_net.parameters(),
                                        weight_decay=self.c_weight_decay,
                                        lr=self.c_lr)

        self.c3_opt = torch.optim.Adam(self.c3_feature_net.parameters(),
                                        weight_decay=self.c_weight_decay,
                                        lr=self.c_lr)
        
    def train(self, source_traindata, target_traindata, split, verbose, PLOT=True):
        """
        Args:
            source_traindata: dfaDataSetTorch
            target_traindata: dfaDataSetTorch
            split: Boolean
            verbose: int
            PLOT: Boolean
        """
        if split:
            logger.info("split the data evenly in to 4")
            train_set = split_train_data(source_traindata, 4)
            source_traindata1 = train_set[0]
            source_traindata2 = train_set[1] 
            source_traindata3 = train_set[2]
            source_traindata4 = train_set[3]
            if self.gpu_flg:
                source_traindata1 = source_traindata1.to_gpu()
                source_traindata2 = source_traindata2.to_gpu()
                source_traindata3 = source_traindata3.to_gpu()
                source_traindata4 = source_traindata4.to_gpu()
                if target_traindata is not None:
                    target_traindata = target_traindata.to_gpu()

            for t in range(self.n_epoch):
                self.cme_w_xc_feature_update(source_traindata2, verbose)
                self.h0_feature_update(source_traindata2, source_traindata3, verbose)
                self.cme_w_x_feature_update(source_traindata1, target_traindata, verbose)
                self.m0_feature_update(source_traindata1, source_traindata4, verbose)

                #self.cme_wc_x_outcome_update(train_data1, train_data2, train_data3, verbose)
                if verbose >= 1:
                    logger.info(f"Epoch {t} ended")    

    
            

        else:
            if self.gpu_flg:
                source_traindata = source_traindata.to_gpu()
            
            for t in range(self.n_epoch):
                self.cme_w_xc_feature_update(source_traindata, verbose)
                self.h0_feature_update(source_traindata, source_traindata, verbose)
                self.cme_w_x_feature_update(source_traindata, target_traindata, verbose)
                self.m0_feature_update(source_traindata, source_traindata, verbose)
                
                if verbose >= 1:
                    logger.info(f"Epoch {t} ended")
        
        if PLOT:
            plt.figure()
            plt.plot(self.track_loss['h0'], label='h0')
            plt.xlabel('iteration')
            plt.ylabel('h0 loss')
            plt.yscale('log')
            plt.savefig('h0_loss.png', bbox_inches='tight')   

            plt.figure()   
            plt.plot(self.track_loss['w_xc'], label='w_xc')
            plt.xlabel('iteration')
            plt.ylabel('w_xc loss')
            plt.yscale('log')
            plt.savefig('cme_w_xc_loss.png', bbox_inches='tight')   

        
            plt.figure()   
            plt.plot(self.track_loss['w_x.x'], label='w_x.x')
            plt.xlabel('iteration')
            plt.ylabel('w_x.x loss')
            plt.yscale('log')
            plt.savefig('cme_w_x_x_source.png', bbox_inches='tight') 

            plt.figure()   
            plt.plot(self.track_loss['w_x.x_target'], label='wc_x.x')
            plt.xlabel('iteration')
            plt.ylabel('w_x.x_target loss')
            plt.yscale('log')
            plt.savefig('cme_w_x_x_target.png', bbox_inches='tight') 
            
            plt.figure()   
            plt.plot(self.track_loss['m0'], label='wc_x.x')
            plt.xlabel('iteration')
            plt.ylabel('m0 loss')
            plt.yscale('log')
            plt.savefig('m0_loss.png', bbox_inches='tight')      

        dpa = DeepPartialGraph(self.x1_source_feature_net, 
                    self.x1_target_feature_net, 
                    self.x2_feature_net,
                    self.x4_feature_net,
                    self.w2_feature_net, 
                    self.c2_feature_net, 
                    self.c3_feature_net,
                    self.add_cme_intercept, 
                    self.add_h0_intercept,
                    self.add_m0_intercept)
        
        #fit thr trained model 
        if split: 
            dpa.fit_t(source_traindata1, 
                      source_traindata2,
                      source_traindata3,
                      source_traindata4,
                      target_traindata, 
                      self.lam_set['cme'], 
                      self.lam_set['h0'],
                      self.lam_set['m0'])
        else:
            dpa.fit_t(source_traindata,
                      source_traindata,
                      source_traindata,
                      source_traindata,
                      target_traindata, 
                      self.lam_set['cme'],
                      self.lam_set['h0'],
                      self.lam_set['m0'])
        return dpa


    # def cme_w_xc_feature_update(self, train_data, verbose):
    #     self.x4_feature_net.train(False)
    #     super(PartialDeepKernelTrainer, self).cme_w_xc_feature_update(train_data, verbose)
     
    
    # def h0_feature_update(self, train_data1, train_data2, verbose):
    #     self.x4_feature_net.train(False)
    #     super(PartialDeepKernelTrainer, self).h0_feature_update(train_data1, train_data2, verbose)



    def m0_feature_update(self, train_data1, train_data2, verbose):
        self.x1_source_feature_net.train(False)

        if self.x1_target_feature_net:
            self.x1_target_feature_net.train(False)
        
        self.x2_feature_net.train(False)
        self.x4_feature_net.train(True)

        self.w2_feature_net.train(False)
        self.c2_feature_net.train(False)

        self.c3_feature_net.train(False)
        
        with torch.no_grad():
            c3_feature2 = self.c3_feature_net(train_data2.C)
            x1_feature1 = self.x1_source_feature_net(train_data1.X)
            x1_feature2 = self.x1_source_feature_net(train_data2.X)
            w2_feature1 = self.w2_feature_net(train_data1.W)
        
        for _ in range(self.m0_iter):
            self.x4_opt.zero_grad()

            x4_feature2 = self.x4_feature_net(train_data2.X)

            res = DeepPartialGraph.fit_m0(x1_feature1,
                                        x1_feature2,
                                        x4_feature2,
                                        w2_feature1,
                                        c3_feature2,
                                        self.lam_set['cme'], 
                                        self.lam_set['m0'],
                                        self.add_cme_intercept,
                                        self.add_m0_intercept)
            
            
            loss = res['loss']
            loss.backward()# update Model2 parameters

            self.track_loss['m0'].append(loss.item())
            if verbose >= 2:
                logger.info(f"m0 learning: {loss.item()}")
            self.x4_opt.step()
            
    def cme_w_x_feature_update(self, source_traindata, target_traindata, verbose=-1):
        """
        Estimate cme(wc|x), update x feature kernel
        """
        self.x1_source_feature_net.train(True)

        if self.x1_target_feature_net:
            self.x1_target_feature_net.train(True)
        
        self.x2_feature_net.train(False)
        self.x4_feature_net.train(False)

        self.w2_feature_net.train(False)
        self.c2_feature_net.train(False)

        self.c3_feature_net.train(False)

        with torch.no_grad():
            w2_feature = self.w2_feature_net(source_traindata.W)
            if target_traindata is not None:
                
                w2_target_feature = self.w2_feature_net(target_traindata.W)
        
        for i in range(self.cme_iter):
            self.x1_source_opt.zero_grad()
            if self.x1_target_feature_net and target_traindata is not None:
                self.x1_target_opt.zero_grad()
                
                x1_target_feature = self.x1_target_feature_net(target_traindata.X)
                x1_target_feature = DeepFullGraph.augment_single_feature(x1_target_feature, self.add_cme_intercept)


            x1_feature = self.x1_source_feature_net(source_traindata.X)
            x1_feature = DeepFullGraph.augment_single_feature(x1_feature, self.add_cme_intercept)



            loss =  linear_reg_loss(w2_feature, x1_feature, self.lam_set['cme'])

            self.track_loss['w_x.x'].append(loss.item())
            
            if self.x1_target_feature_net and target_traindata is not None:
                l2 = linear_reg_loss(w2_target_feature, x1_target_feature, self.lam_set['cme'])
                loss += l2
                self.track_loss['w_x.x_target'].append(l2.item())

            loss.backward()
            
            
            if verbose >= 2:
                logger.info(f"cme_w_x x learning: {loss.item()}")

            self.x1_source_opt.step()
            if self.x1_target_feature_net:
                self.x1_target_opt.step()



    