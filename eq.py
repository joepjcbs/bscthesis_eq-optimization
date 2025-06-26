import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, butter, freqz
from matplotlib.ticker import ScalarFormatter

class EQ():
    """
    A multiband equalizer using Butterworth filters.

    Parameters:
        fs (int): Sampling frequency.
        freqs (list): Center frequencies for the filters.
        mod_indices (list of lists): Indices of filters that will be adjusted dynamically.
    """
    def __init__(self, fs, freqs, mod_indices):
        self.fs = fs
        self.freqs = freqs 
        self.n_freqs = len(freqs)
        self.mod_indices = mod_indices
        self.gains = np.zeros(self.n_freqs)
        self.filter_bank = self.__create_filter_bank()

    def __create_filter_bank(self):
        """
        Creates a filter bank based on the defined center frequencies.
        Each band is either low-pass, high-pass, or band-pass.

        Returns:
            list: A list of filter coefficient tuples (b, a).
        """

        filters = list()
        for band_idx in range(self.n_freqs):
            freq = int(self.freqs[band_idx])
            filter_type = ""
            if band_idx == 0:
                filter_type = "low"
            elif band_idx == self.n_freqs - 1:
                filter_type = "high"
            else:
                filter_type = "band"
            
            filters.append(self.__create_filter(freq, self.gains[band_idx], filter_type))

        return filters
    
    def __create_filter(self, frequency, gain_db, filter_type="band", Q=2):
        """
        Create a Butterworth filter for a specific band with gain.

        Parameters:
            frequency (float): Center frequency.
            gain_db (float): Gain in decibels.
            filter_type (str): Type of filter: 'low', 'high', or 'band'.
            Q (float): Q-factor for band-pass filters, which determines the bandwith.

        Returns:
            tuple: Filter coefficients (b, a).
        """

        # Convert decibels to linear factor
        gain_linear = 10**(gain_db / 20)

        # Create filters based on type
        if filter_type == 'low':
            b, a = butter(7, frequency / (self.fs / 2), btype='low', analog=False)
        elif filter_type == 'high':
            b, a = butter(7, frequency / (self.fs / 2), btype='high', analog=False)
        else:  # Band-pass
            bandwidth = frequency / Q
            low = max(1, frequency - bandwidth / 2)
            high = min(self.fs / 2 - 1, frequency + bandwidth / 2)
            b, a = butter(4, [low / (self.fs / 2), high / (self.fs / 2)], btype='band',  analog=False)
        return b * gain_linear, a
    
    def update_filter_gains(self, new_gains):
        """
        Update the gains of modulated frequency bands and rebuild the filter bank.

        Parameters:
            new_gains (list): List of new gain values in dB.
        """
        for idx, new_gain in zip(self.mod_indices, new_gains):
            for i in idx:
                self.gains[i] = new_gain
        self.filter_bank = self.__create_filter_bank()
    
    def apply_eq(self, audio_np, n_channels):
        """
        Applies the EQ to multi-channel audio data.

        Parameters:
            audio_np (np.ndarray): Audio data, shape (samples, channels).
            n_channels (int): Number of channels.

        Returns:
            np.ndarray: Equalized audio data in int16 format.
        """
        # Initialize the output buffer
        filtered_audio = np.zeros_like(audio_np, dtype=np.float64)

        # Filter each band and sum them per channel
        for b, a in self.filter_bank:
            for ch_idx in range(n_channels):
                band_filtered = filtfilt(b, a, audio_np[:, ch_idx].astype(np.float64))
                filtered_audio[:, ch_idx] += band_filtered

        # Soft clipping to avoid distortion
        threshold = 32767
        filtered_audio = np.tanh(filtered_audio / threshold) * threshold * 0.90

        # Convert back to int16 for playback
        filtered_audio = filtered_audio.astype(np.int16)

        return filtered_audio

    def plot_eq_response(self):
        """
        Plot the frequency response of the current filter bank.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        for b, a in self.filter_bank:
            w, h = freqz(b, a, worN=512, fs=self.fs)
            ax.plot(w, 20 * np.log10(abs(h)))

        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_ylim(-20, 20)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title('EQ Frequency Coverage')
        ax.axhline(y=0, color='black', linestyle='--')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.show()
    
