import torch.nn as nn

class trucles_linear(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(trucles_linear, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        
        self.relu = nn.ReLU()
        
        #self.tanh = nn.Tanh()
        self.dense4 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dense1(x)
        #x = self.tanh(x)
        x = self.relu(x)

        x = self.dense4(x)
        x = self.sigmoid(x)
        return x

class trucles_linear_deep(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(trucles_linear_deep, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim*4)
        
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.relu1 = nn.ReLU()
        self.dense3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu2 = nn.ReLU()
        
        #self.tanh = nn.Tanh()
        self.dense4 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dense1(x)
        #x = self.tanh(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu1(x)
        x = self.dense3(x)
        x = self.relu2(x)
        x = self.dense4(x)
        x = self.sigmoid(x)
        return x
        
class trucles_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(trucles_lstm, self).__init__()
        
        self.blstm1 = nn.LSTM(input_dim, hidden_dim, bidirectional = True,
                              dropout = 0.1, batch_first=True, num_layers=2)
        self.blstm2 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True,
                              dropout = 0.2, batch_first=True, num_layers=2)

        self.dense2 = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        #print(x.shape)
        x, (h0,c0) = self.blstm1(x)
        x, (h0,c0) = self.blstm2(x)
       # x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x 