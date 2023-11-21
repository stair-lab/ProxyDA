#Author: Katherine Tsai <kt14@illinois.edu>
#MIT License

import jax.numpy as jnp
import numpy as np
from lsa import generate_multienv_data, tidy_w



def gen_multienv_class_discrete_z(z_env, seed, num_samples, partition_dict = {'train': 0.7, 'val': 0.1, 'test': 0.2}):


    samples_dict = generate_multienv_data(z_indicator=z_env, 
                                          seed=seed, 
                                          num_samples=num_samples, 
                                          partition_dict=partition_dict)

    samples_dict_tidy = tidy_w(samples_dict, w_value=3)
    keys = samples_dict_tidy['train'].keys()

    source_train = {}
    #source_val   = {}
    source_test  = {}

    Maps = {'x':'X', 'w':'W', 'c':'C', 'u':'U', 'y_one_hot':'Y_one_hot', 'y': 'Y'}
    for key in keys:
        if key in Maps.keys():
            source_train[Maps[key]] = jnp.array(samples_dict_tidy['train'][key])
            #source_val[Maps[key]]   = jnp.array(samples_dict_tidy['val'][key])
            source_test[Maps[key]]  = jnp.array(samples_dict_tidy['test'][key]) 
    
    source_train['Z'] = jnp.ones(source_train['X'].shape[0])*z_env
    #source_val['Z']   = jnp.ones(source_val['X'].shape[0])*z_env
    source_test['Z']  = jnp.ones(source_test['X'].shape[0])*z_env
    
    z = np.zeros((source_train['X'].shape[0], samples_dict_tidy['train']['n_env']))
    z[:, z_env] = 1.
    source_train['Z_one_hot'] = jnp.array(z)

    #z = np.zeros((source_val['X'].shape[0],  samples_dict_tidy['val']['n_env']))
    #z[:, z_env] = 1.
    #source_val['Z_one_hot'] = jnp.array(z)
    
    z = np.zeros((source_test['X'].shape[0], samples_dict_tidy['test']['n_env']))
    z[:, z_env] = 1.
    source_test['Z_one_hot'] = jnp.array(z)


    return source_train, source_test


        
    
    


