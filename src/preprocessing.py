import numpy as np
from scipy.signal import butter, filtfilt, detrend, stft

def preprocess_ppg(ppg_signal, fs=1000):
    """ Membersihkan sinyal PPG dan mengekstrak fitur STFT (Spectrogram) """
    # Bandpass Filter & Baseline Correction
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)
    corrected = detrend(filtered, type='linear')
    norm = (corrected - np.mean(corrected)) / (np.std(corrected) + 1e-8)
    
    # Transformasi ke Time-Frequency menggunakan STFT
    # nperseg=128 akan menghasilkan 65 bins frekuensi (channel)
    f, t, Zxx = stft(norm, fs=fs, nperseg=128, noverlap=64)
    
    # Ambil nilai Magnitude (karena output asli STFT adalah bilangan kompleks)
    magnitude_spectrogram = np.abs(Zxx)
    
    # Normalisasi Min-Max agar AI lebih mudah memproses pola spektrum
    mag_min = np.min(magnitude_spectrogram)
    mag_max = np.max(magnitude_spectrogram)
    mag_norm = (magnitude_spectrogram - mag_min) / (mag_max - mag_min + 1e-8)
    
    # Output sekarang adalah Matriks 2D dengan shape (65, Time_Steps)
    return mag_norm