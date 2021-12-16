import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import NoisyLinear
from networks.network_bodies import SimpleBody, AtariBody

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5, body=SimpleBody):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy

        self.body = body(input_shape, num_actions, noisy, sigma_init)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(self.body.feature_size(),
                                                                                               512, sigma_init)
        self.fc2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(512, self.num_actions,
                                                                                       sigma_init)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()