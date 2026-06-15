import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import preprocess_ppg
from sklearn.model_selection import train_test_split

class PPGClassificationDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
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

        for idx in range(len(ppg_data)):
            # Memanggil fungsi preprocessing (STFT + Z-Score)
            processed = preprocess_ppg(ppg_data[idx], fs=1000)
            sbp, dbp = labels_data[idx]
            
            if sbp < 130 and dbp < 80:
                label_class = 0 
            elif (130 <= sbp < 140) or (80 <= dbp < 90):
                label_class = 1 
            else:
                label_class = 2 
            
            signals.append(processed)
            labels.append(label_class)

    class_counts = np.array([labels.count(0), labels.count(1), labels.count(2)])
    total_samples = len(labels)
    
    balanced_weights = total_samples / (3.0 * class_counts)
    class_weights_tensor = torch.tensor(balanced_weights, dtype=torch.float32)
    
    print(f"\n[INFO] Bobot Hukuman AI: Normal={balanced_weights[0]:.2f}, Hyp I={balanced_weights[1]:.2f}, Hyp II={balanced_weights[2]:.2f}\n")
    
    test_val_ratio = val_split + test_split
    train_signals, temp_signals, train_labels, temp_labels = train_test_split(
        signals, labels, test_size=test_val_ratio, stratify=labels, random_state=42
    )

    test_ratio_from_temp = test_split / test_val_ratio
    val_signals, test_signals, val_labels, test_labels = train_test_split(
        temp_signals, temp_labels, test_size=test_ratio_from_temp, stratify=temp_labels, random_state=42
    )

    train_dataset = PPGClassificationDataset(train_signals, train_labels)
    val_dataset = PPGClassificationDataset(val_signals, val_labels)
    test_dataset = PPGClassificationDataset(test_signals, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights_tensor