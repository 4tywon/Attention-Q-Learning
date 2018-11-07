import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#allow hardware acceleration if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DQN(nn.Module):

    def __init__(self, **kwargs):
        super(DQN, self).__init__()
        self.stateLength = 100
        self.hiddenDim = 100
        if "stateLength" in kwargs:
            self.stateLength = kwargs["stateLength"]
        if "hiddenDim" in kwargs:
            self.hiddenDim = kwargs["hiddenDim"]

        self.actionDim = 100
        self.objectDim = 100
        if "actionDim" in kwargs:
            self.actionDim = kwargs["actionDim"]
        if "objectDim" in kwargs:
            self.objectDim = kwargs["objectDim"]
            
        self.initial = nn.Linear(self.stateLength, self.hiddenDim) 
        self.objects = nn.Linear(self.hiddenDim, self.objectDim)
        self.actions = nn.Linear(self.hiddenDim, self.actionDim)
    def forward(self, x):
        x = F.relu(self.initial(x))
        y = self.objects(x)
        z = self.actions(x)
        return z,y
