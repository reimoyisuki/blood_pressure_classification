import torch
import torch.nn as nn

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, in_channels=65):
        super(CNN_LSTM_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64) 
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 3)    

    def forward(self, x):
        # Melewati Blok 1
        x = self.relu(self.bn1(self.conv1(x)))
        # Melewati Blok 2 lalu di-pooling
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        last_state = lstm_out[:, -1, :] 
        
        out = self.fc(self.dropout(last_state))
        return out