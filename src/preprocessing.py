import numpy as np
from scipy.signal import butter, filtfilt, detrend

def preprocess_ppg(ppg_signal, fs=125):
    """ Membersihkan sinyal PPG dengan Bandpass Filter dan Baseline Correction """
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)
    corrected = detrend(filtered, type='linear')
    norm = (corrected - np.mean(corrected)) / (np.std(corrected) + 1e-8)
    return norm