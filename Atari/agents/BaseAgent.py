import numpy as np
import pickle
import os.path
import csv
import pickle

import torch
import torch.optim as optim


class BaseAgent(object):
    def __init__(self, config, env, log_dir='/tmp'):
        self.model=None
        self.target_model=None
        self.optimizer = None

        self.log_dir = log_dir

        self.rewards = []

        self.action_log_frequency = config.ACTION_SELECTION_COUNT_FREQUENCY
        self.action_selections = [0 for _ in range(env.action_space)]

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def MSE(self, x):
        return 0.5 * x.pow(2)

    def save_w(self):
        torch.save(self.model.state_dict(), 'model.dump')
        torch.save(self.optimizer.state_dict(), 'optim.dump')

    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/exp_replay_agent.dump', 'wb'))

    def load_replay(self):
        fname = './saved_agents/exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_+= torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            
            if count > 0:
                with open(os.path.join(self.log_dir, 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_/count))

    def save_reward(self, reward):
        self.rewards.append(reward)
        with open('rewards.pickle', 'wb') as handle:
            pickle.dump(self.rewards, handle)


    def save_action(self, action, tstep):
        self.action_selections[int(action)] += 1.0/self.action_log_frequency
        if (tstep+1) % self.action_log_frequency == 0:
            with open(os.path.join(self.log_dir, 'action_log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list([tstep]+self.action_selections))
            self.action_selections = [0 for _ in range(len(self.action_selections))]
