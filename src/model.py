import torch
import torch.nn as nn

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, in_channels=65): # 65 adalah jumlah bin frekuensi dari STFT
        super(CNN_LSTM_Classifier, self).__init__()
        
        # CNN membaca pola frekuensi sepanjang waktu
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        # LSTM memproses urutan waktu hasil ekstraksi CNN
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        
        # Dropout untuk mencegah AI menghafal (Overfitting)
        self.dropout = nn.Dropout(0.3)
        
        # Output layer untuk 3 Kelas (0 = Normal, 1 = Hipertensi I, 2 = Hipertensi II)
        self.fc = nn.Linear(128, 3)    

    def forward(self, x):
        # x input shape: (batch, 65, time_steps)
        x = self.pool(self.relu(self.conv1(x)))
        
        # Transpose untuk LSTM -> (batch, time_steps, features)
        x = x.permute(0, 2, 1)  
        
        lstm_out, _ = self.lstm(x)
        last_state = lstm_out[:, -1, :] # Ambil kesimpulan di akhir waktu
        
        # Masukkan ke linear layer dengan dropout
        out = self.fc(self.dropout(last_state))
        return out