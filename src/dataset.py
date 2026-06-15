import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.preprocessing import preprocess_ppg
from sklearn.model_selection import train_test_split

class PPGClassificationDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Karena output STFT sudah 2D (65, time_steps), gausa unsqueeze(0) lagi
        signal = torch.tensor(self.signals[idx], dtype=torch.float32) 
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

def get_classification_loaders(data_path, batch_size=16, val_split=0.15, test_split=0.15):
    patients = sorted(list(set([fn.split("_")[0] for fn in os.listdir(data_path) if fn.endswith("_ppg.npy")])))
    signals, labels = [], []

    for patient in patients:
        ppg_fn = os.path.join(data_path, patient + "_ppg.npy")
        labels_fn = os.path.join(data_path, patient + "_labels.npy")
        
        if not os.path.exists(ppg_fn) or not os.path.exists(labels_fn):
            continue
            
        ppg_data = np.load(ppg_fn)
        labels_data = np.load(labels_fn)

        if len(ppg_data) == 0 or len(labels_data) == 0:
            continue

        for idx in range(len(ppg_data)):
            processed = preprocess_ppg(ppg_data[idx], fs=1000)
            sbp, dbp = labels_data[idx]
            
            # RULE KLASIFIKASI 3 RENTANG
            if sbp < 130 and dbp < 80:
                label_class = 0 # Normal
            elif (130 <= sbp < 140) or (80 <= dbp < 90):
                label_class = 1 # Hipertensi I
            else:
                label_class = 2 # Hipertensi II
            
            signals.append(processed)
            labels.append(label_class)

    print("="*40)
    print(" STATISTIK DATA KESELURUHAN (POPULASI) ")
    print("="*40)
    print(f"Total Segmen Sinyal : {len(signals)}")
    print(f"Normal (0)          : {labels.count(0)}")
    print(f"Hipertensi I (1)    : {labels.count(1)}")
    print(f"Hipertensi II (2)   : {labels.count(2)}")
    print("="*40)
    
    test_val_ratio = val_split + test_split
    train_signals, temp_signals, train_labels, temp_labels = train_test_split(signals, labels, test_size=test_val_ratio, stratify=labels, random_state=42)

    test_ratio_from_temp = test_split / test_val_ratio
    val_signals, test_signals, val_labels, test_labels = train_test_split(temp_signals, temp_labels, test_size=test_ratio_from_temp, stratify=temp_labels, random_state=42)

    train_dataset = PPGClassificationDataset(train_signals, train_labels)
    val_dataset = PPGClassificationDataset(val_signals, val_labels)
    test_dataset = PPGClassificationDataset(test_signals, test_labels)

    # Hitung jumlah tiap kelas di data training
    class_counts = np.array([train_labels.count(0), train_labels.count(1), train_labels.count(2)])
    
    # Berikan bobot yang lebih besar untuk kelas yang minoritas
    class_weights = 1.0 / (class_counts + 1e-8)
    
    # Tempelkan bobot tersebut ke masing-masing sampel
    samples_weights = np.array([class_weights[t] for t in train_labels])
    samples_weights = torch.from_numpy(samples_weights)
    
    # Buat sampler (AI akan dipaksa mengambil kelas 1 dan 2 lebih sering agar seimbang)
    sampler = WeightedRandomSampler(weights=samples_weights.type('torch.DoubleTensor'), num_samples=len(samples_weights), replacement=True)

    # Masukkan sampler ke DataLoader (PENTING: shuffle harus False jika memakai sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader