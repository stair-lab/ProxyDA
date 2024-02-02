import numpy as np
import scipy
from scipy.ndimage import rotate
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skimage.measure import block_reduce
from KPLA.data.dSprite.gen_data_wpc import U2imgs, img2X, XU2C, CU2Y, get_rot_mat



def generate_n_simplex(n):
  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)  



def sample_beta(row, U_dists):
#     row = row if row < N_ENVS + 4 else 1
    alpha, beta = U_dists[row[0]]
    return np.array([np.random.beta(alpha, beta)])


def Z2U(Z, U_dists):
    U = np.apply_along_axis(lambda x: sample_beta(x, U_dists), axis=1, arr=Z).reshape(-1,1)
    
    U = 2. * np.pi * U
    return U

def U2W(U, pos_X_basis, pos_Y_basis):
    N = U.shape[0]
    U_W = np.zeros((N, 1))
    for i in tqdm(range(N), desc='getting U rotation matrix'):
        rot_mat = get_rot_mat(U[i][0])
        U_W[i][0] = 10 * (rot_mat @ np.array(
            [[pos_X_basis], [pos_Y_basis]]))[1, 0] # pos_Y

    return np.random.normal(U_W, 0.25)

def CU2Y_v2(C, U, pos_X_basis, pos_Y_basis, task='regression'):
  var = np.random.normal(0, 0.1, size=U.shape)# * (U/(2*np.pi))

  N = U.shape[0]
  U_Y = np.zeros((N, 1))

  for i in tqdm(range(N), desc='getting U rotation matrix'):
    rot_mat = get_rot_mat(U[i][0])
    U_Y[i][0] = (rot_mat @ np.array(
        [[pos_X_basis], [pos_Y_basis]]))[1, 0] # pos_Y

  return (5*C*U_Y) + var


def generate_samples_Z2U(Z, A, metadata, pos_X_basis, pos_X_basis_idx,
                     pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis,
                      n_samples=10000, test_size=0.3, task='regression', dom=0,
                        target=False, alpha_2=0.6, N_ENVS=4, U_dists=[], one_hot=True):

    if target:
      U = 2. * np.pi * np.random.uniform(alpha_2, 1, size=(Z.shape[0],1))
    else:
      U = Z2U(Z, U_dists)

    imgs_sampled = U2imgs(U, metadata, pos_X_basis_idx, pos_Y_basis_idx, imgs, imgs_basis)

    X = img2X(imgs_sampled).reshape(U.shape[0], -1)
    C = XU2C(X, U, pos_X_basis, pos_Y_basis, A)
    Y = CU2Y(C, U, pos_X_basis, pos_Y_basis, task=task)
    W = U2W(U, pos_X_basis, pos_Y_basis)

    if not target:
      Z = OneHotEncoder(categories=[list(range(N_ENVS))]).fit_transform(Z).toarray()

    (X_train, X_val,
     Z_train, Z_val,
     Y_train, Y_val,
     W_train, W_val,
     U_train, U_val,
     imgs_train, imgs_val) = train_test_split(
        X, Z, Y, W, U, imgs_sampled,
        test_size=test_size,
        shuffle=True,
    )

    (X_val, X_test,
     Z_val, Z_test,
     Y_val, Y_test,
     W_val, W_test,
     U_val, U_test,
     imgs_val, imgs_test) = train_test_split(
        X_val, Z_val, Y_val, W_val, U_val, imgs_val,
        test_size=test_size,
        shuffle=True,
    )
    #if not one_hot:
    #  Z_train = Z_train.flatten()+1
    #  Z_test  = Z_test.flatten()+1
    #  Z_val   = Z_val.flatten()+1
    train = {
        'X': X_train, 'Z': Z_train, 'Y': Y_train, 'W': W_train, 'U': U_train,
        # 'orig_X': X_train,
    }
    val = {
        'X': X_val, 'Z': Z_val, 'Y': Y_val, 'W': W_val, 'U': U_val,
        # 'orig_X': X_val,
    }
    test = {
        'X': X_test,'Z': Z_test, 'Y': Y_test, 'W': W_test, 'U': U_test,
        # 'orig_X': X_test,
    }

    # X = X.reshape(-1, 64, 64)

    # pool_size = (1, 8, 8)

    # train = {
    #     'X': block_reduce(X_train, pool_size, np.mean).reshape(X_train.shape[0], -1), 'Z': Z_train, 'Y': Y_train, 'W': W_train, 'U': U_train,
    #     # 'orig_X': X_train,
    # }
    # val = {
    #     'X': block_reduce(X_val, pool_size, np.mean).reshape(X_val.shape[0], -1), 'Z': Z_val, 'Y': Y_val, 'W': W_val, 'U': U_val,
    #     # 'orig_X': X_val,
    # }
    # test = {
    #     'X': block_reduce(X_test, pool_size, np.mean).reshape(X_test.shape[0], -1), 'Z': Z_test, 'Y': Y_test, 'W': W_test, 'U': U_test,
    #     # 'orig_X': X_test,
    # }

    return train, val, test, {
        'train': imgs_train, 'val': imgs_val, 'test': imgs_test,
    }



def generate_samples_Z2U_v2(Z, A, metadata, pos_X_basis, pos_X_basis_idx,
                     pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis,
                      n_samples=10000, test_size=0.3, task='regression', dom=0,
                        target=False, alpha_2=0.6, N_ENVS=4, U_dists=[]):

    if target:
      U = 2. * np.pi * np.random.uniform(alpha_2, 1, size=(Z.shape[0],1))
    else:
      U = np.apply_along_axis(lambda x: sample_beta(x, U_dists), axis=1, arr=Z).reshape(-1,1)
    
      U = (2. * np.pi * U)/4+(2. * np.pi*Z)/4

    imgs_sampled = U2imgs(U, metadata, pos_X_basis_idx, pos_Y_basis_idx, imgs, imgs_basis)

    X = img2X(imgs_sampled).reshape(U.shape[0], -1)
    C = XU2C(X, U, pos_X_basis, pos_Y_basis, A)
    Y = CU2Y(C, U, pos_X_basis, pos_Y_basis, task=task)
    Y = CU2Y(C, U, pos_X_basis, pos_Y_basis, task=task)
    W = U2W(U, pos_X_basis, pos_Y_basis)

    #if not target:
    #  Z = OneHotEncoder(categories=[list(range(N_ENVS))]).fit_transform(Z).toarray()

    (X_train, X_val,
     Z_train, Z_val,
     Y_train, Y_val,
     W_train, W_val,
     U_train, U_val,
     imgs_train, imgs_val) = train_test_split(
        X, Z, Y, W, U, imgs_sampled,
        test_size=test_size,
        shuffle=True,
    )

    (X_val, X_test,
     Z_val, Z_test,
     Y_val, Y_test,
     W_val, W_test,
     U_val, U_test,
     imgs_val, imgs_test) = train_test_split(
        X_val, Z_val, Y_val, W_val, U_val, imgs_val,
        test_size=test_size,
        shuffle=True,
    )

    train = {
        'X': X_train, 'Z': Z_train, 'Y': Y_train, 'W': W_train, 'U': U_train,
        # 'orig_X': X_train,
    }
    val = {
        'X': X_val, 'Z': Z_val, 'Y': Y_val, 'W': W_val, 'U': U_val,
        # 'orig_X': X_val,
    }
    test = {
        'X': X_test,'Z': Z_test, 'Y': Y_test, 'W': W_test, 'U': U_test,
        # 'orig_X': X_test,
    }

    # X = X.reshape(-1, 64, 64)

    # pool_size = (1, 8, 8)

    # train = {
    #     'X': block_reduce(X_train, pool_size, np.mean).reshape(X_train.shape[0], -1), 'Z': Z_train, 'Y': Y_train, 'W': W_train, 'U': U_train,
    #     # 'orig_X': X_train,
    # }
    # val = {
    #     'X': block_reduce(X_val, pool_size, np.mean).reshape(X_val.shape[0], -1), 'Z': Z_val, 'Y': Y_val, 'W': W_val, 'U': U_val,
    #     # 'orig_X': X_val,
    # }
    # test = {
    #     'X': block_reduce(X_test, pool_size, np.mean).reshape(X_test.shape[0], -1), 'Z': Z_test, 'Y': Y_test, 'W': W_test, 'U': U_test,
    #     # 'orig_X': X_test,
    # }

    return train, val, test, {
        'train': imgs_train, 'val': imgs_val, 'test': imgs_test,
    }