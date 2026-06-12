import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import preprocess_ppg

class PPGClassificationDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Format untuk CNN1D: (batch, channel, seq_len)
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0) 
        # Target klasifikasi butuh format LongTensor (Integer)
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
            processed = preprocess_ppg(ppg_data[idx])
            sbp, dbp = labels_data[idx]
            
            # RULE KLASIFIKASI: 1 = Hipertensi, 0 = Normal / Pre
            label_class = 1 if (sbp >= 140 or dbp >= 90) else 0
            
            signals.append(processed)
            labels.append(label_class)

    print(f"Total sinyal berhasil dimuat: {len(signals)}")
    
    total_data = len(signals)
    train_ratio = 1.0 - val_split - test_split
    train_split_idx = int(train_ratio * total_data)
    val_split_idx = int(val_split * total_data)

    train_signals, train_labels = signals[:train_split_idx], labels[:train_split_idx]
    val_signals, val_labels = signals[train_split_idx : train_split_idx + val_split_idx], labels[train_split_idx : train_split_idx + val_split_idx]
    test_signals, test_labels = signals[train_split_idx + val_split_idx :], labels[train_split_idx + val_split_idx :]

    train_dataset = PPGClassificationDataset(train_signals, train_labels)
    val_dataset = PPGClassificationDataset(val_signals, val_labels)
    test_dataset = PPGClassificationDataset(test_signals, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader