import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from toeplitzlda.classification import ToeplitzLDA
import pyxdf

from auditory_aphasia.bscthesis_eq.eq import EQ
from auditory_aphasia.bscthesis_eq.session_manager import Session
from auditory_aphasia.bscthesis_eq.random_search import RandomSearch

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')


def load_and_preprocess_raw(header_file, filter_band=(0.5, 16)):
    """
    Load BrainVision EEG data, apply filtering, montage, and pick EEG channels.

    Parameters:
        header_file (str or Path): Path to the .vhdr file.
        filter_band (tuple): Bandpass filter range (low, high) in Hz.

    Returns:
        mne.io.Raw: Preprocessed raw EEG data.
    """
    non_eeg_channels = ["EOGvu", "x_EMGl", "x_GSR", "x_Respi", "x_Pulse", "x_Optic"]
    raw = mne.io.read_raw_brainvision(header_file, misc=non_eeg_channels, preload=True)
    raw.set_montage("standard_1020")
    raw.filter(*filter_band, method="iir")
    raw.pick_types(eeg=True)
    return raw

def load_and_preprocess_xdf(xdf_file, filter_band):
    """
    Load EEG + marker data from XDF file and convert to MNE Raw object.

    Parameters:
        xdf_file (str or Path): Path to the XDF file.
        filter_band (tuple): Bandpass filter range (low, high) in Hz.

    Returns:
        mne.io.Raw: Preprocessed raw EEG data with annotations.
    """

    data, _ = pyxdf.load_xdf(xdf_file, dejitter_timestamps=False)

    # Get stream names
    stream_names = [stream['info']['name'][0] for stream in data]

    # Find EEG stream
    idx_eeg = stream_names.index("BrainAmp BUA -")
    eeg_stream = data[idx_eeg]

    # Transpose to shape (channels, samples)
    eeg_data = np.array(eeg_stream['time_series']).T
    ts_eeg = np.array(eeg_stream['time_stamps'])
    info_eeg = eeg_stream['info']

    # Find Marker stream
    idx_markers = stream_names.index("AuditoryAphasiaAudioMarker")
    markers = [int(m[0]) for m in data[idx_markers]['time_series']]
    time_stamps = np.array(data[idx_markers]['time_stamps'])

    # Create info for MNE
    ch_names = [ch['label'][0] for ch in info_eeg['desc'][0]['channels'][0]['channel']]
    sfreq = float(info_eeg['nominal_srate'][0])
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    onsets = time_stamps - ts_eeg[0]

    # Create RawArray
    raw = mne.io.RawArray(eeg_data, info)

    annotations = mne.Annotations(
        onset=onsets,
        duration=[0] * len(time_stamps),
        description=markers
    )

    raw.set_annotations(annotations)

    if 'MkIdx' in raw.ch_names:
        raw.set_channel_types({'MkIdx': 'stim'})

    # Preprocessing
    raw.set_montage("standard_1020")
    raw.filter(*filter_band, method="iir")
    raw.pick_types(eeg=True)

    return raw

def epoch_raw(raw, decimate=10):
    """
    Epoch raw data into events and categorize them as targets or non-targets.

    Parameters:
        raw (mne.io.Raw): Preprocessed EEG recording.
        decimate (int): Decimation factor to reduce data size.

    Returns:
        mne.Epochs: Epoched and labeled data.
    """

    id_map = {
        '101': 101,
        '102': 102,
        '103': 103,
        '104': 104,
        '105': 105,
        '106': 106,
        '111': 111,
        '112': 112,
        '113': 113,
        '114': 114,
        '115': 115,
        '116': 116,
        '201': 201,
        '202': 202,
        '203': 203,
        '204': 204,
        '205': 205,
        '206': 206,
    }

    evs, _ = mne.events_from_annotations(raw, event_id=id_map)
    target_ids = list(range(111, 117))
    non_target_ids = list(range(101, 107))
    event_id = {f"Word_{i - 110}/Target": i for i in target_ids}
    event_id.update({f"Word_{i - 100}/NonTarget": i for i in non_target_ids})
    epoch = mne.Epochs(raw, events=evs, event_id=event_id, decim=decimate,
                       proj=False, tmax=1, baseline=None)
    return epoch

def load_data(session, filter_band=(0.5, 16)):
    """
    Load, preprocess, and segment EEG data from all XDF runs for a subject.

    Parameters:
        session (Session): The current experiment session manager.
        filter_band (tuple): Bandpass filter range in Hz.

    Returns:
        tuple: (all_epochs, iterations, trials)
            - all_epochs (mne.Epochs): Combined raw epochs.
            - iterations (list): Groups of 6 epochs.
            - trials (list): Groups of 15 iterations (90 epochs).
    """

    # Get the data path
    data_dir = session.run_folder_path / f"sub-{session.subject_name}" / "ses-S001" / "eeg"

    xdf_files = sorted(data_dir.glob(f"sub-{session.subject_name}_ses-S001_task-Default_run-*_eeg.xdf"))

    if not xdf_files:
        print(f"No XDF files found in {data_dir}")
        return []

    # Load the data, preprocess and slice it into epochs
    epochs = list()
    for xdf_file in xdf_files:
        session.logger.debug(f"Loading data of {xdf_file.name}")
        raw_data = load_and_preprocess_xdf(xdf_file, filter_band)
        epochs.append(epoch_raw(raw_data))

    epochs = mne.concatenate_epochs(epochs)

    # Combine 6 epochs into a single iteration
    iterations = [epochs[i:i+6] for i in np.arange(0, epochs.events.shape[0],6)]

    # Assert that each iteration contains exactly 1 Target
    assert all([len(iteration["Target"]) == 1 for iteration in iterations]), "Number of targets in single iterations is unequal to 1."

    # Combine 15 iterations into a trial
    trials = [iterations[i:i+15] for i in np.arange(0,len(iterations),15)]

    return epochs, iterations, trials

def get_jumping_means(epo, boundaries):
    """
    Compute mean signal values within successive time windows (jumping means).

    Parameters:
        epo (mne.Epochs): Epochs to compute features from.
        boundaries (list): List of time boundaries in seconds.

    Returns:
        np.ndarray: Array of shape (n_epochs, n_channels, n_windows).
    """

    shape_orig = epo.get_data().shape
    X = np.zeros((shape_orig[0], shape_orig[1], len(boundaries)-1))
    for i in range(len(boundaries)-1):
        idx = epo.time_as_index((boundaries[i], boundaries[i+1]))
        idx_range = list(range(idx[0], idx[1]))
        X[:,:,i] = epo.get_data()[:,:,idx_range].mean(axis=2)
    return X

def compute_auc_per_config(trials, clf_ival_boundaries):
    """
    Compute AUC scores for each trial using ToeplitzLDA and stratified CV.

    Parameters:
        trials (list): List of trial objects (each with 15 iterations of 6 stimuli).
        clf_ival_boundaries (list): Time intervals for computing jumping mean features.

    Returns:
        np.ndarray: AUC score per trial.
    """

    results = np.zeros(len(trials))
    for t, trial in enumerate(trials):
        features, labels = list(), list()
        for iteration in trial:
            for s, stimulus in enumerate(iteration):
                features.extend([
                    get_jumping_means(iteration[s], clf_ival_boundaries).squeeze().flatten()
                ])
            labels.extend([
                1 if event > 107 else 0
                for event in iteration.events[:,2]
            ])
        X = np.array(features)
        print(f"X shape: {X.shape}")
        y = np.array(labels)
        n_channels = trial[0].get_data().shape[1]
        clf = ToeplitzLDA(n_channels=n_channels)
        skf = StratifiedKFold(n_splits=15, shuffle=False)
        auc_scores = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')
        results[t] = auc_scores.mean()

    return results

def run_analysis(session: Session):
    """
    Full pipeline: loads data, computes features, performs classification,
    and saves AUC scores per trial to disk.

    Parameters:
        session (Session): Active session with data and logging context.

    Returns:
        np.ndarray: Array of AUC scores per trial.
    """

    print('Loading data...')
    epochs, _, trials = load_data(session)

    clf_ival_boundaries = [0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]
    session.logger.info(f"Classification intervals: {clf_ival_boundaries}")

    auc_scores = compute_auc_per_config(trials, clf_ival_boundaries)
    session.logger.info(f"auc_shape {auc_scores.shape}")
    session.logger.info(f"Mean AUC: {np.mean(auc_scores)}")
    
    np.save(session.run_folder_path / 'auc_per_config.npy', auc_scores)
    session.logger.info('Saved AUC scores...')

    return auc_scores