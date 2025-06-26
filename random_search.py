import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=8)

class RandomSearch():
    """
    Random search algorithm for sampling gain configurations across EQ bands.

    Parameters:
        n_bands (int): Number of EQ bands (dimensions of the search space).
        min_gain (float): Minimum gain (in dB).
        max_gain (float): Maximum gain (in dB).
        min_dist (float): Minimum distance required between sampled configurations.
    """

    def __init__(self, n_bands, min_gain = -6.0, max_gain = 6.0, min_dist = 2.0):
        self.n_bands = n_bands
        self.history = list()
        self.min_dist = min_dist
        self.min_gain = min_gain
        self.max_gain = max_gain

    def set_random_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Parameters:
            seed (int): Random seed value.
        """

        np.random.seed(seed)

    def __check_resample(self, sample):
        """
        Check whether the given sample is too close to any existing sample.

        Parameters:
            sample (np.ndarray): New gain configuration.

        Returns:
            bool: True if the sample is too close and should be resampled.
        """

        for config in self.history:
            dist = np.linalg.norm(sample - config)
            if dist < self.min_dist:
                print('Resampling...')
                return True
        
        return False

    def sample_gains(self):
        """
        Sample a new gain configuration that is sufficiently distant from previous ones.

        Returns:
            np.ndarray: Sampled gain values (1D array of length n_bands).
        """

        while True:
            gains = np.zeros(self.n_bands)

            for idx_band in range(self.n_bands):
                gains[idx_band] = np.random.uniform(self.min_gain, self.max_gain)

            if not self.__check_resample(gains):
                break

        self.history.append(gains)
        return gains

    def simulate_n_samples(self, n_samples):
        """
        Generate multiple gain configurations for simulation/testing.

        Parameters:
            n_samples (int): Number of configurations to sample.
        """

        for i in range(n_samples):
            self.sample_gains()
    
    def clear_history(self):
        """
        Clear the history of sampled configurations.
        """

        self.history = list()