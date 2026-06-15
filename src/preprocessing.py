import numpy as np
from scipy.signal import butter, filtfilt, detrend, stft

def preprocess_ppg(ppg_signal, fs=1000):
    """ Membersihkan sinyal PPG dan mengekstrak spektrum Log-STFT """
    # 1. Bandpass Filter & Baseline Correction
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)
    corrected = detrend(filtered, type='linear')
    norm_sig = (corrected - np.mean(corrected)) / (np.std(corrected) + 1e-8)
    
    # 2. Transformasi ke Time-Frequency menggunakan STFT
    f, t, Zxx = stft(norm_sig, fs=fs, nperseg=128, noverlap=64)
    magnitude = np.abs(Zxx)
    
    # 3. Log-Transform (SANGAT KRUSIAL: Membuka kontras frekuensi sinyal)
    log_spectrogram = np.log1p(magnitude) # ln(1 + x)
    
    # 4. Normalisasi Min-Max untuk Jaringan Saraf
    mag_min = np.min(log_spectrogram)
    mag_max = np.max(log_spectrogram)
    mag_norm = (log_spectrogram - mag_min) / (mag_max - mag_min + 1e-8)
    
    return mag_norm