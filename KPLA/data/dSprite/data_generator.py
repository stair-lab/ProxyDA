"""Generates data samples for the dSprite regression task."""

# Author: Olawale Salaudeen <oes2@illinois.edu> Nicole Chiou <nicchiou@stanford.edu>

import numpy as np

from KPLA.data.dSprite.gen_data_wpc import latent_to_index, generate_samples


def generate_data(
    path, N, alpha_1, beta_1, dist_1, alpha_2, beta_2, dist_2, verbose=False
):
    results = {}

    dataset_zip = np.load(
        path + "/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        allow_pickle=True,
        encoding="bytes",
    )

    if verbose:
        print("Keys in the dataset:", dataset_zip.keys())
    imgs = dataset_zip["imgs"]
    metadata = dataset_zip["metadata"][()]
    if verbose:
        print("Metadata: \n", metadata)

    # Define number of values per latents and functions to convert to indices
    U_basis = np.zeros((3, 6))
    for i in range(3):
        U_basis[i, 1] = i
        U_basis[i, 2] = 5
        U_basis[i, -2] = 16

    pos_X_basis_idx, pos_Y_basis_idx = 16, 0

    pos_X_basis = (
        metadata[b"latents_possible_values"][b"posX"][pos_X_basis_idx] - 0.5
    )
    pos_Y_basis = (
        metadata[b"latents_possible_values"][b"posY"][pos_Y_basis_idx] - 0.5
    )

    indices_basis = latent_to_index(U_basis, metadata)
    imgs_basis = imgs[indices_basis]

    # Exploration
    A = np.random.uniform(0, 1, size=(10, 4096))

    # Generate Data
    u_params = {
        "source_dist": dist_1,
        "source_dist_params": {"alpha": alpha_1, "beta": beta_1},
        "target_dist": dist_2,
        "target_dist_params": {"min": alpha_2, "max": beta_2},
    }

    results["u_params"] = u_params

    if dist_1 == "uniform":
        U_s = np.random.uniform(alpha_1, beta_1, size=(N, 1)) * 2 * np.pi
    elif dist_1 == "beta":
        U_s = np.random.beta(alpha_1, beta_1, size=(N, 1)) * 2 * np.pi
    else:
        raise NotImplementedError()

    if dist_2 == "uniform":
        U_t = np.random.uniform(alpha_2, beta_2, size=(N, 1)) * 2 * np.pi
    elif dist_2 == "beta":
        U_t = np.random.beta(alpha_2, beta_2, size=(N, 1)) * 2 * np.pi
    else:
        raise NotImplementedError()

    print("SOURCE")
    source_train, source_val, source_test, source_imgs_dict = generate_samples(
        U_s,
        A,
        metadata,
        pos_X_basis,
        pos_X_basis_idx,
        pos_Y_basis,
        pos_Y_basis_idx,
        imgs,
        imgs_basis,
    )

    print("TARGET")
    target_train, target_val, target_test, target_imgs_dict = generate_samples(
        U_t,
        A,
        metadata,
        pos_X_basis,
        pos_X_basis_idx,
        pos_Y_basis,
        pos_Y_basis_idx,
        imgs,
        imgs_basis,
    )

    results["source_train"] = source_train
    results["source_val"] = source_val
    results["source_test"] = source_test
    results["source_imgs_dict"] = source_imgs_dict

    results["target_train"] = target_train
    results["target_val"] = target_val
    results["target_test"] = target_test
    results["target_imgs_dict"] = target_imgs_dict

    return results
