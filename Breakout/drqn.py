import numpy as np
import matplotlib.pyplot as plt
# import gym
import torch
from torch import nn
from breakout import Breakout
import random
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import sys

np.set_printoptions(threshold=sys.maxsize)

class DRQN(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim, n_layers, batch_size, seq_len, gamma=0.9):
        super(DRQN, self).__init__()
        self.actions = action_dim  # 3
        self.obs_dim = observation_dim  # 6
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # we are doing gradient descent WRT optimizer corresponding q
        self.SEQ_LEN = seq_len
        self.BATCH_SIZE = batch_size
        self.N_EPOCHS = 50
        self.gamma = gamma
        
        # input: (batch size, seq_len, obs_dim)
        # output: (batch size, seq_len, hidden_dim)
        self.rnn = nn.RNN(self.obs_dim, hidden_dim, n_layers, batch_first=True).float()
        # input: (*, hidden_dim),(*, action_dim)
        self.fc = nn.Linear(hidden_dim, action_dim).float()

    def forward(self, x):
        # print("*********************************************")
        # print("enter the forward func, x.shape", x.shape)   # ([50, 6])
        curr_batch_size = x.size(0)
        curr_sequence_length = x.size(1)
        x = x.view(curr_batch_size, curr_sequence_length, -1)
        hidden = torch.zeros(1, curr_batch_size, self.hidden_dim, dtype=torch.float)
        # print("size of x:", x.shape)    # 1,6,1
        out, hidden = self.rnn(x, hidden)
        # print("size of out:", out.shape)
        # print("size of hidden:", hidden.shape)
        x = self.fc(out)
        # print("size of x after linear layer:", x.shape, x)    # 1,6,1
        # print("*********************************************")
        return x, hidden


class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length=sequence_length

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # print("entering sample", len(self.memory), batch_size)      # 70, 50
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x-self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            #correct for sampling near beginning
            final = self.memory[max(start+1,0):end+1]
            
            #correct for sampling across episodes
            for i in range(len(final)-2, -1, -1):
                if final[i][3] is None:
                    final = final[i+1:]
                    break
                    
            #pad beginning to account for corrections
            while(len(final)<self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]))] + final                            
            samp+=final

        #returns flattened version
        # print(len(samp), samp[0])
        return samp

    def __len__(self):
        return len(self.memory)