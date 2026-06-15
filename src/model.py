import torch
import torch.nn as nn

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, in_channels=65):
        super(CNN_LSTM_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        # TAMBAHAN: BatchNorm agar AI tidak pusing melihat skala matriks STFT
        self.bn1 = nn.BatchNorm1d(64) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 3)    

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.permute(0, 2, 1)  
        
        lstm_out, _ = self.lstm(x)
        last_state = lstm_out[:, -1, :] 
        
        out = self.fc(self.dropout(last_state))
        return out