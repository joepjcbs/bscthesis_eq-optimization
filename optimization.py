import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from itertools import product

from auditory_aphasia.bscthesis_eq.session_manager import Session

def fit_gp(configurations, auc_scores):
    """
    Fit a Gaussian Process to configuration vs. AUC data.

    Parameters:
        configurations (np.ndarray): Array of gain configurations (shape: [n_samples, n_features]).
        auc_scores (np.ndarray): Corresponding AUC scores.

    Returns:
        gp (GaussianProcessRegressor): Trained Gaussian Process.
        auc_scores_gp (np.ndarray): GP-predicted AUC values for input configurations.
    """

    kernel =  RBF(10.0, (2.5, 10.0)) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(configurations, auc_scores)

    auc_scores_gp = gp.predict(configurations)
    return gp, auc_scores_gp

def gp_slice(gp, fixed_gain3=0.0, min_db=-6.0, max_db=6.0, resolution=50):
    """
    Slice the GP's 3D prediction surface at a fixed gain_3 to get a 2D surface.

    Parameters:
        gp (GaussianProcessRegressor): Trained GP.
        fixed_gain3 (float): Value of gain_3 to fix.
        min_db (float): Minimum gain in dB.
        max_db (float): Maximum gain in dB.
        resolution (int): Number of steps along gain_1 and gain_2.

    Returns:
        G1, G2 (np.ndarray): Meshgrid of gain_1 and gain_2.
        Z (np.ndarray): Predicted AUC scores over the 2D slice.
    """

    # Create a 2D grid of gain_1 and gain_2 values
    g1 = np.linspace(min_db, max_db, resolution)
    g2 = np.linspace(min_db, max_db, resolution)
    G1, G2 = np.meshgrid(g1, g2)

    # Flatten and stack with fixed gain_3 to form the grid for GP
    grid_points = np.c_[G1.ravel(), G2.ravel(), np.full(G1.size, fixed_gain3)]

    # Predict AUC using the GP
    y_pred, y_std = gp.predict(grid_points, return_std=True)
    Z = y_pred.reshape(G1.shape)

    return G1, G2, Z

def plot_gp(gp, min_db = -6.0, max_db = 6.0,):
    """
    Plot 2D slices of the 3D GP prediction over combinations of gain_1 and gain_2
    while sweeping gain_3 across a fixed range.

    Parameters:
        gp (GaussianProcessRegressor): Trained GP.
        min_db (float): Minimum dB for gain values.
        max_db (float): Maximum dB for gain values.
    """

    g3_range = np.linspace (min_db, max_db, 25)

    G1_list = list()
    G2_list = list()
    Z_list = list()

    # For each fixed gain_3 value, compute a 2D GP prediction slice
    for g3_idx in range(len(g3_range)):
        fixed_g3 = g3_range[g3_idx]
        G1, G2, Z = gp_slice(gp, fixed_gain3 = fixed_g3)
        G1_list.append(G1)
        G2_list.append(G2)
        Z_list.append(Z)

    vmin = min(Z.min() for Z in Z_list)
    vmax = max(Z.max() for Z in Z_list)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])  

    fig, axs = plt.subplots(5, 5, figsize=(15,15))
    axs = axs.flatten()

    contours = list()

    # Loop over each subplot and draw the contour map for the corresponding slice
    for i, Z in enumerate(Z_list):
        ax = axs[i]
        contour = ax.contourf(G1_list[i], G2_list[i], Z, levels=20, vmax=1, vmin=0)
        contours.append(contour)
        ax.set_title(f"Gain 3 = {g3_range[i]:.2f}dB")

    fig.supxlabel("Gain 1 (dB)")
    fig.supylabel("Gain 2 (dB)")
    fig.suptitle("Predicted AUC scores of gain 1 and 2 over slices of gain 3")
    cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), pad=8, label="Predicted AUC")
    cbar.ax.set_position([0.9, 0.15, 0.02, 0.7])
    plt.subplots_adjust(right=0.88, hspace=0.3)
    plt.show()

def sample_grid(n_bands = 3, min_db =-6.0, max_db = 6.0, resolution=50):
    """
    Generate a grid of possible gain configurations.

    Parameters:
        n_bands (int): Number of EQ bands.
        min_db (float): Minimum gain.
        max_db (float): Maximum gain.
        resolution (int): Number of samples per band.

    Returns:
        np.ndarray: All possible configurations in the grid.
    """

    values = np.linspace(min_db, max_db, resolution)
    grid = np.array(list(product(values, repeat=n_bands)))
    return grid

def sample_best_config(gp,  k=5, min_distance=1.5):
    """
    Sample the top-k predicted configurations from the GP model,
    enforcing a minimum Euclidean distance between them.

    Parameters:
        gp (GaussianProcessRegressor): Trained GP.
        k (int): Number of configurations to sample.
        min_distance (float): Minimum allowed distance between selected configs.

    Returns:
        np.ndarray: Array of shape (k, n_features) with the top-k configurations.
    """

    grid = sample_grid()
    auc_scores = gp.predict(grid)
    sorted_indices = np.argsort(auc_scores)[::-1]

    selected = []
    for idx in sorted_indices:
        candidate = grid[idx]
        if all(np.linalg.norm(candidate - sel) >= min_distance for sel in selected):
            selected.append(candidate)
            if len(selected) == k:
                break

    return np.array(selected)

def run_optimization(session: Session):
    """
    Run Gaussian Process-based optimization using stored session data.

    Parameters:
        session (Session): Session object with run path and logger.

    Returns:
        np.ndarray: Best configuration predicted by the GP.
    """
    auc_scores = np.load(session.run_folder_path / 'auc_per_config.npy')
    configs = np.load(session.run_folder_path / 'configs.npy')

    gp, auc_scores_gp = fit_gp(configs, auc_scores)
    session.logger.info("Fit GP")
    plot_gp(gp)
    best_configs_gp = sample_best_config(gp)
    session.logger.info(f"Top 5: {best_configs_gp}")
    best_config_gp = best_configs_gp[0]
    session.logger.info(f"Best configuration: {best_config_gp}")
    return best_config_gp