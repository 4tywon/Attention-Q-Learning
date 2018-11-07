import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim


class LSTM_mp(nn.Module):
    def __init__(self, hiddenDim, inputDim, seqLength, batchSize):
        super(LSTM_mp, self).__init__()
        self.hiddenDim = hiddenDim
        self.inputDim = inputDim
        self.batchSize = batchSize
        self.LSTM = nn.LSTM(inputDim, hiddenDim)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # hidden state and cell state
        return (autograd.Variable(
            torch.zeros(1,self.batchSize,self.hiddenDim)),
                autograd.Variable(
                    torch.zeros(1,self.batchSize,self.hiddenDim)))
    def forward(self, x):
                # x has shape (seq_len, batch, input_size)
        out, self.hidden = self.LSTM(x, self.hidden)
        return torch.mean(out, 1)

    
