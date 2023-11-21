"""
    Cross-validation pipeline for the adaptation methods
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import numpy as np
import jax.numpy as jnp
from itertools import product
from models.plain_kernel.adaptation import full_adapt
from models.plain_kernel.multienv_adaptation import multienv_adapt, multienv_adapt_categorical


def tune_adapt_modelCV(source_train:  dict, 
                        target_train: dict,
                        source_test:  dict,
                        target_test:  dict,
                        method_set: dict, 
                        kernel_dict: dict, 
                        model, #function class object
                        task='r',
                        n_params=5, 
                        n_fold=5,
                        min_log=-4,
                        max_log=4,):

    best_estimator = None
    best_err = np.inf if task=='r' else -np.inf
    
    params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                     np.logspace(min_log, max_log, n_params).tolist(), #alpha2
                     np.logspace(min_log, max_log, n_params).tolist()) #scale

    best_params = {}

    for alpha, alpha2, scale in params:
        kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
        lam_set = {'cme': alpha, 'k0': alpha2, 'h0': alpha2, 'lam_min':-4, 'lam_max':0}
        errs = []
        best_err_i = np.inf if task=='r' else -np.inf
        best_model_i = None



        for i, (train_idx, test_idx) in enumerate(kf.split(source_train['X'])):
            print(f"({alpha}, {alpha2}, {scale}), Fold {i}:")

            source_train_cv_train = {k: v[train_idx] for k, v in source_train.items()} 
            source_train_cv_val   = {k: v[test_idx]  for k, v in source_train.items()} 

            split = False

            estimator = model(source_train_cv_train, target_train, source_test,
                              target_test, split, scale, lam_set, method_set, kernel_dict)
            estimator.fit(task)
            ##select parameters from source
            predictY = estimator.predict({'X': source_train_cv_val['X']}, 
                                        'source', 
                                        'source')
            if task == 'r':
                acc_err = mean_squared_error(np.array(source_train_cv_val['Y']), np.array(predictY))
            
            elif task == 'c':
                testY_label = np.array(jnp.argmax(source_train_cv_val['Y'], axis=1))
                predictY_prob = normalize(np.array(predictY), axis=1)
                
                acc_err = roc_auc_score(testY_label, predictY_prob[:,1])
                

            
            errs.append(acc_err/len(source_train_cv_val))
            ## select parameters from target
            improve_r = (acc_err < best_err_i) and task == 'r'
            improve_c = (acc_err > best_err_i) and task == 'c'
            if improve_r or improve_c:
                best_err_i = acc_err
                best_model_i = estimator

        improve_r = (np.mean(errs) < best_err) and task == 'r'
        improve_c = (np.mean(errs) > best_err) and task == 'c'
        
        if improve_r or improve_c:
            best_err = np.mean(errs)
            best_estimator = best_model_i
            best_params = {'alpha':alpha, 'alpha2':alpha2, 'scale':scale}
           
            print(f"update best parameters alpha: {alpha}, alpha2:{alpha2}, scale: {scale}, err: {np.mean(errs)}\n")

    return best_estimator, best_params



def tune_adapt_model(source_train: dict, 
                     target_train: dict,
                     source_test: dict, 
                     target_test: dict,
                     source_val:   dict,
                     method_set:   dict, 
                     kernel_dict:   dict, 
                     task='r',
                     n_params=5, 
                     min_log=-4,
                     max_log=4,
                     ):

    best_estimator = None
    best_err = np.inf if task=='r' else -np.inf
    
    params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                     np.logspace(min_log, max_log, n_params).tolist()) #scale

    best_params = {}

    for alpha, scale in params:
        lam_set = {'cme': alpha, 'k0': alpha, 'h0': alpha, 'lam_min':-4, 'lam_max':0}
        errs = []


        print(f"({alpha}, {scale}):")

        split = False

        estimator = full_adapt(source_train, target_train, source_test,
                          target_test, split, scale, lam_set, method_set, kernel_dict)
        estimator.fit(task)
        ##select parameters from source
        predictY = estimator.predict({'X': source_val['X']}, 
                                    'source', 
                                    'source')
        if task == 'r':
            acc_err = mean_squared_error(np.array(source_val['Y']), np.array(predictY))
        
        elif task == 'c':
            testY_label = np.array(jnp.argmax(source_val['Y'], axis=1))
            predictY_prob = normalize(np.array(predictY), axis=1)
            
            acc_err = roc_auc_score(testY_label, predictY_prob[:,1])
                

        
        improve_r = (acc_err < best_err) and task == 'r'
        improve_c = (acc_err > best_err) and task == 'c'
        
        if improve_r or improve_c:        

            best_err = acc_err
            best_estimator = estimator
            best_params = {'alpha':alpha, 'scale':scale}
           
            print(f"update best parameters alpha: {alpha}, scale: {scale}, err: {acc_err}\n")

    return best_estimator, best_params





def tune_multienv_adapt_model(source_train_list: list, 
                              target_train_list: list,
                              source_test_list:  list, 
                              target_test_list:  list,
                              source_val_list:   list,
                              method_set:        dict, 
                              kernel_dict:       dict, 
                              models = multienv_adapt,
                              task='r',
                              n_params=5, 
                              min_log=-4,
                              max_log=4,
                            ):

    best_estimator = None
    best_err = np.inf if task=='r' else -np.inf
    
    params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                     np.logspace(min_log, max_log, n_params).tolist()) #scale

    best_params = {}

    for alpha, scale in params:
        lam_set = {'cme': alpha, 'k0': alpha, 'h0': alpha, 'lam_min':-4, 'lam_max':0}
        errs = []


        print(f"({alpha}, {scale}):")

        split = False

        estimator = models(source_train_list, 
                          target_train_list, 
                          source_test_list,
                          target_test_list, 
                          split, 
                          scale, 
                          lam_set, 
                          method_set, 
                          kernel_dict)
                          
        estimator.fit(task)
        ##select parameters from source
        acc_err = 0
        for source_val in source_val_list:
            predictY = estimator.predict({'X': source_val['X']}, 
                                        'source', 
                                        'source')
            
            if task == 'r':
                acc_err += mean_squared_error(np.array(source_val['Y']), 
                                             np.array(predictY))
            
            elif task == 'c':
                testY_label = np.array(jnp.argmax(source_val['Y'], axis=1))
                predictY_prob = normalize(np.array(predictY), axis=1)
                
                acc_err += roc_auc_score(testY_label, predictY_prob[:,1])
                

        
        improve_r = (acc_err < best_err) and task == 'r'
        improve_c = (acc_err > best_err) and task == 'c'
        
        if improve_r or improve_c:        

            best_err = acc_err
            best_estimator = estimator
            best_params = {'alpha':alpha, 'scale':scale}
           
            print(f"update best parameters alpha: {alpha}, scale: {scale}, err: {acc_err}\n")

    return best_estimator, best_params



def tune_multienv_adapt_modelCV(source_train_list:  dict, 
                                target_train_list: dict,
                                source_test_list:  dict,
                                target_test_list:  dict,
                                method_set: dict, 
                                kernel_dict: dict, 
                                model, #function class object
                                task='c',
                                n_params=5, 
                                n_fold=5,
                                min_log=-4,
                                max_log=4):

    best_estimator = None
    best_err = np.inf if task=='r' else -np.inf
    
    params = product(np.logspace(min_log, max_log, n_params).tolist(), #alpha
                     np.logspace(min_log, max_log, n_params).tolist(), #alpha2
                     np.logspace(min_log, max_log, n_params).tolist()) #scale

    best_params = {}

    for alpha, alpha2, scale in params:
        kf = KFold(n_splits=n_fold, random_state=None, shuffle=False)
        lam_set = {'cme': alpha, 'k0': alpha2, 'h0': alpha2, 'lam_min':-4, 'lam_max':0}
        errs = []
        best_err_i = np.inf if task=='r' else -np.inf
        best_model_i = None


        for i, (train_idx, test_idx) in enumerate(kf.split(source_train_list[0]['X'])):
            print(f"({alpha}, {alpha2}, {scale}), Fold {i}:")

            source_train_cv_train = [{k: v[train_idx] for k, v in st.items()} for st in source_train_list] 
            source_train_cv_val   = [{k: v[test_idx]  for k, v in st.items()} for st in source_train_list] 

            split = False

            estimator = multienv_adapt_categorical(source_train_cv_train, 
                              target_train_list, 
                              source_test_list,
                              target_test_list,
                              split, 
                              scale, 
                              lam_set, 
                              method_set, 
                              kernel_dict)


            estimator.fit(task=task)
            acc_err = 0
            ##select parameters from source
            for id, source_val in enumerate(source_train_cv_val):
                predictY = estimator.predict({'X': source_val['X']}, 
                                            'source', 
                                            'source', id)
                if task == 'r':
                    acc_err += mean_squared_error(np.array(source_val['Y']), np.array(predictY))
                
                elif task == 'c':
                    testY_label = np.array(jnp.argmax(source_val['Y'], axis=1))
                    predictY_prob = normalize(np.array(predictY), axis=1)
                    
                    acc_err += roc_auc_score(testY_label, predictY_prob[:,1])
                

            
            errs.append(acc_err/len(source_train_cv_val))
            ## select parameters from target
            improve_r = (acc_err < best_err_i) and task == 'r'
            improve_c = (acc_err > best_err_i) and task == 'c'
            if improve_r or improve_c:
                best_err_i = acc_err
                best_model_i = estimator

        improve_r = (np.mean(errs) < best_err) and task == 'r'
        improve_c = (np.mean(errs) > best_err) and task == 'c'
        
        if improve_r or improve_c:
            best_err = np.mean(errs)
            best_estimator = best_model_i
            best_params = {'alpha':alpha, 'alpha2':alpha2, 'scale':scale}
           
            print(f"update best parameters alpha: {alpha}, alpha2:{alpha2}, scale: {scale}, err: {np.mean(errs)}\n")

    return best_estimator, best_params