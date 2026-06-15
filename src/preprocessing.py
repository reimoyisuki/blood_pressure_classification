import numpy as np
from scipy.signal import butter, filtfilt, detrend, stft

def preprocess_ppg(ppg_signal, fs=1000):
    """ Membersihkan sinyal PPG dan mengekstrak spektrum Log-STFT """
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)
    corrected = detrend(filtered, type='linear')
    norm_sig = (corrected - np.mean(corrected)) / (np.std(corrected) + 1e-8)
    
    f, t, Zxx = stft(norm_sig, fs=fs, nperseg=128, noverlap=64)
    magnitude = np.abs(Zxx)
    
    log_spectrogram = np.log1p(magnitude) # ln(1 + x)
    
    mag_norm = (log_spectrogram - np.mean(log_spectrogram)) / (np.std(log_spectrogram) + 1e-8)
    
    return mag_norm