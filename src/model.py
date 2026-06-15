import torch
import torch.nn as nn

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        
        # Output layer untuk 3 Kelas (0 = Normal, 1 = Hipertensi I, 2 = Hipertensi II)
        self.fc = nn.Linear(128, 3)    

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)  # Ubah ke (batch, seq_len, features) untuk LSTM
        
        lstm_out, _ = self.lstm(x)
        last_state = lstm_out[:, -1, :] # Ambil state terakhir dari sekuens
        
        out = self.fc(last_state)
        return out