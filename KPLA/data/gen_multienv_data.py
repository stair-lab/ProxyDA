#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from jax import random
import numpy as np
import jax.numpy as jnp


def gen_Z(n, mean, sigma, key):
    Z = (random.normal(key,(n,))*sigma)+mean

    return Z


def gen_Zcategorical(n, prob, key):
    Z = random.choice(key, jnp.arange(4), (n,), p=np.array(prob))

    return Z

def gen_Ucategorical(Z, n, key):

    e1 = random.uniform(key[0],(n,),minval=0,maxval=0.25)+Z*0.25
    U2 = 3*random.uniform(key[1],(n,),minval=0,maxval=1)+np.where((Z>1), 0, -1)
    e3= np.where((U2>1),0,-1)
    e4= np.where((U2<0),0,-1)
    e5=(e3+e4)
    U1=e1+e5+1
    
    return U1, U2


def gen_U(Z, n, key):
    e1=random.uniform(key[0],(n,),minval=0,maxval=1)
    U2=3*random.uniform(key[1],(n,),minval=0,maxval=1)-1
    e3= np.where((Z>0.5),0,-1)
    e4= np.where((Z<-0.5),0,-1)
    e5=(e3+e4)
    U1=e1+e5+1

    return U1, U2


def gen_X(U1,U2, m_x, v_x, n,  key):
    X1= U1+ (random.normal(key[0],(n,))*v_x[0])+m_x[0]
    X2= U2+ random.uniform(key[1],(n,),minval=-1,maxval=1)

    return X1, X2


def gen_W(U1,U2, m_w, v_w, n,  key):
    W1= U1+ random.uniform(key[0],(n,),minval=-1,maxval=1)
    W2= U2+ (random.normal(key[1],(n,)) *v_w[1])+m_w[1]

    return W1, W2


def gen_Y(X1, X2, U1, U2, n):
  mask =  (U1*U2 > 0)
  d = mask.astype(int)*2-1

  y= U2*(np.cos(2*(X1*X2+.3*U1+.2))+d)
  return y


###

def from_Z_to_U_cat(z_indicator):
  pu_lookup = {1:[0.8, 0.1, 0.1], 2:[0.1, 0.8, 0.1], 3:[0.1, 0.1, 0.8], 
               4:[0.3, 0.7, 0.0], 5:[0., 0.3, 0.7], 6:[0.3, 0.2, 0.5]}

  return pu_lookup[z_indicator]

def gen_U_cat(keys, prob, n):
    U1 = random.bernoulli(keys[0], p=prob[0], shape=(n,1))*2-1
    U2 = random.bernoulli(keys[1], p=prob[1], shape=(n,1))*2-1
    U3 = random.bernoulli(keys[2], p=prob[2], shape=(n,1))
    return U1, U2, U3

def gen_X_from_U_cat(keys, U1, U2, U3, m_x, v_x, n):    
    X1= m_x[0]*U1 + (2*U3-1)*m_x[0]/2 + (random.normal(keys[0],(n,)) *v_x[0])
    X2= U2*(jnp.cos(X1) + U3*m_x[1]+(U1+1)/2) + random.normal(keys[1],(n,))*v_x[1]
    X3 = U3*jnp.sin(X2) + U3*U1 + random.normal(keys[2], (n,))*v_x[2]
    return X1, X2, X3


def gen_W_from_U_cat(keys, U1, U2, U3, m_w, v_w, n):
    W1= U1* random.uniform(keys[0],(n,), minval=-1, maxval=0) + U3+(random.normal(keys[1],(n,)) *v_w[0])
    W2= U1*jnp.sin(m_w[1]*U2) + U1*random.uniform(keys[0],(n,), minval=0, maxval=1)
    W3 = U1*U3*3 + random.normal(keys[2], (n,))*v_w[1]

    return W1, W2, W3

def gen_Y_from_UX_cat(X1, X2, X3, U1, U2, U3, n):
    mask = (U2*X1)>0
    Y = jnp.sin(X2+mask) + (U3*2-1)*jnp.cos(X3) 
    return Y




