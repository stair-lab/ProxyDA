import numpy as np
from scipy.ndimage import rotate
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

def crop_center(img, cropx, cropy):
  y, x, *_ = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)    
  return img[starty:starty + cropy, startx:startx + cropx, ...]

def latent_to_index(latents, metadata):
  latents_sizes = metadata[b'latents_sizes']
  latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                              np.array([1,])))
  return np.dot(latents, latents_bases).astype(int)

def sample_latent(metadata, size=1, p_dict=None):
  latents_sizes = metadata[b'latents_sizes']
  samples = np.zeros((size, latents_sizes.size))
  for lat_i, lat_size in enumerate(latents_sizes):
    if p_dict is None:
      samples[:, lat_i] = np.random.randint(lat_size, size=size)
    else:
      samples[:, lat_i] = np.random.choice(range(lat_size), p=p_dict[lat_i], size=size)

  return samples

def get_rot_mat(theta):
  c, s = np.cos(theta), np.sin(theta)
  R = np.array(((c, -s), (s, c)))
  return R

def x_trans(x, A):
  z = np.linalg.norm(x@A.T, axis=1, keepdims=True)**2
  z /= 10
  z -= 5000
  z /= 2000
  
  return z
  
def U2imgs(U, metadata, pos_X_basis_idx, pos_Y_basis_idx, imgs, imgs_basis):
  N = U.shape[0]
  img_basis = np.random.randint(3, size=N)
  latents = np.zeros((N, 6))
  latents[:, 1] = 2 # img_basis #shape
  latents[:, 2] = 5 # fix the scale - replace this with U (6)
  latents[:, -2] = pos_X_basis_idx #xpos (32)
  latents[:, -1] = pos_Y_basis_idx #ypos (32)

  latents = latents.astype(int)
  indices = latent_to_index(latents, metadata)
  imgs_sampled = imgs[indices]

  dx, dy = imgs_sampled.shape[-2], imgs_sampled.shape[-1]
  U_ = np.random.normal(U, 0.25, size=U.shape)
  for i in tqdm(range(N), desc='applying U image rotation'):
    imgs_sampled[i] = crop_center(
        rotate(imgs_basis[latents[i][1]], np.rad2deg(U_[i][0])),
        dx, dy)

  return imgs_sampled
  
def img2X(img_samples):
  N, d, _ = img_samples.shape
  return (
      img_samples +
      np.random.multivariate_normal(np.zeros(d), 0.001*np.eye(d), size=(N, d))
  )

def XU2C(X, U, pos_X_basis, pos_Y_basis, A):
  N = X.shape[0]
  U_C = np.zeros((N, 1))
#     for i in tqdm(range(N), desc='getting U rotation matrix'):
#         rot_mat = get_rot_mat(U[i][0])
#         U_C[i][0] = 10.*(rot_mat @ np.array(
#             [[pos_X_basis], [pos_Y_basis]]))[0, 0] # pos_X
  U_C = U
  X_C = x_trans(X, A)

  C = np.random.normal(X_C**2 + U_C, 0.5)
  return C
  
def CU2Y(C, U, pos_X_basis, pos_Y_basis, task='regression'):
  var = np.random.normal(0, 0.1, size=U.shape)# * (U/(2*np.pi))

  N = U.shape[0]
  U_Y = np.zeros((N, 1))

  for i in tqdm(range(N), desc='getting U rotation matrix'):
    rot_mat = get_rot_mat(U[i][0])
    U_Y[i][0] = (rot_mat @ np.array(
        [[pos_X_basis], [pos_Y_basis]]))[1, 0] # pos_Y

  return (5*C + U_Y) / 20 + var

def U2W(U, pos_X_basis, pos_Y_basis):
  N = U.shape[0]
  U_W = np.zeros((N, 1))
#     U_W = U
  for i in tqdm(range(N), desc='getting U rotation matrix'):
    rot_mat = get_rot_mat(U[i][0])
    U_W[i][0] = 10 * (rot_mat @ np.array(
        [[pos_X_basis], [pos_Y_basis]]))[0, 0] # pos_X

  return np.random.normal(U_W, 0.25)

def generate_samples(U, A, metadata, pos_X_basis, pos_X_basis_idx,
                     pos_Y_basis, pos_Y_basis_idx, imgs, imgs_basis,
                      n_samples=10000, test_size=0.3, task='regression'):

  imgs_sampled = U2imgs(U,
                        metadata,
                        pos_X_basis_idx,
                        pos_Y_basis_idx,
                        imgs,
                        imgs_basis)

  X = img2X(imgs_sampled).reshape(U.shape[0], -1)
  C = XU2C(X, U, pos_X_basis, pos_Y_basis, A)
  Y = CU2Y(C, U, pos_X_basis, pos_Y_basis, task=task)
  W = U2W(U, pos_X_basis, pos_Y_basis)

  (X_train, X_val,
    C_train, C_val,
    Y_train, Y_val,
    W_train, W_val,
    U_train, U_val,
    imgs_train, imgs_val) = train_test_split(
      X, C, Y, W, U, imgs_sampled,
      test_size=test_size,
      shuffle=True,
  )

  (X_val, X_test,
    C_val, C_test,
    Y_val, Y_test,
    W_val, W_test,
    U_val, U_test,
    imgs_val, imgs_test) = train_test_split(
      X_val, C_val, Y_val, W_val, U_val, imgs_val,
      test_size=test_size,
      shuffle=True,
  )
  train = {
      'X': X_train, 'C': C_train, 'Y': Y_train, 'W': W_train, 'U': U_train
  }
  val = {
      'X': X_val, 'C': C_val, 'Y': Y_val, 'W': W_val, 'U': U_val
  }
  test = {
      'X': X_test, 'C': C_test, 'Y': Y_test, 'W': W_test, 'U': U_test
  }

  return train, val, test, {
      'train': imgs_train, 'val': imgs_val, 'test': imgs_test
  }
