import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
from breakout import Breakout
from drqn import DRQN, RecurrentExperienceReplayMemory
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import sys

np.set_printoptions(threshold=sys.maxsize)

class DRQNModel():
    def __init__(self, obs_dim, act_dim, hidden_dim, n_layers,batch_size,seq_len, env,gamma=0.9):
        self.observation_dim = obs_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.gamma=gamma
        self.q = DRQN(observation_dim=obs_dim,action_dim=act_dim,
                    hidden_dim=hidden_dim, n_layers=n_layers,batch_size=batch_size,
                    seq_len=seq_len, gamma=gamma)
        self.q_target = DRQN(observation_dim=obs_dim,action_dim=act_dim,
                    hidden_dim=hidden_dim, n_layers=n_layers,batch_size=batch_size,
                    seq_len=seq_len, gamma=gamma)
        self.q_target.load_state_dict(self.q.state_dict())
        self.update_count = 0
        self.target_net_update_freq = 50
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=1e-4)
        self.nstep_buffer = []
        self.nsteps = 100
        self.memory = RecurrentExperienceReplayMemory(capacity=70,sequence_length=seq_len)
        self.reset_hx()
        self.env = env
        # after which frame do we start to sample minibatch
        self.learn_start = 1000

    def reset_hx(self):
        self.seq = [np.zeros(self.observation_dim) for j in range(self.seq_len)]

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))
        if(len(self.nstep_buffer)<self.nsteps):
            return
        R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)
            self.memory.push((state, action, R, None))

    def compute_target(self, next_states, rewards):
        """
        next_states: they torch.Tensor, (batch size, seq_len, obs_dim)
        rewards: torch.Tensor, (batch, 1) ???

        return torch.Tensor, (batch, 1), 1-step Q learning target
        s' to Q(s', a'; w-) and choose the max one in the output tensor
        y_i<- r_i + gamma*max_a'(Q(s', a'; w-))
        the shape of result should be the same as rewards
        """
        max_next_q_values = torch.zeros((self.batch_size, self.seq_len), dtype=torch.float)
        with torch.no_grad():
            # print("@@@@@@@@@@ next states @@@@@@@")
            # print(next_states.shape)
            max_next,_ = self.q_target(next_states)
            max_next_q_values = max_next.max(dim=2)[0]
            y = rewards + ((self.gamma)*max_next_q_values)
        return y

    def loss(self, states, actions, target):
        """
        states: (batch size, seq_len, obs_dim)
        actions: (batch, 1) ???
        target: (batch, 1) ???
        returns (1) with squared Q error
        """
        tmp = self.q(states)[0]
        curr_q_values = tmp.gather(2, actions).squeeze()
        res = nn.functional.mse_loss(curr_q_values, target, reduction='mean')
        return res

    def get_action(self, state):
        """
        states: np.array of size (obs_dim,) with the current state
        returns int as the optimal action
        """
        with torch.no_grad():
            self.seq.pop(0)
            self.seq.append(state)
            X = torch.tensor([self.seq], dtype=torch.float)
            a, _ = self.q(X)
            a = a[:, -1, :]  #select last element of seq
            a = a.max(1)[1]
            return a.item()

    def prep_minibatch(self):
        transitions = self.memory.sample(self.batch_size)
        # 500 samples, but should be batch_size * seq_len=50*15 = 450
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)   # already next state, don't need to sample batch+1
        # print("get initial", len(batch_state), len(batch_next_state))
        shape = (self.batch_size,self.seq_len)+(self.observation_dim,)

        batch_state = torch.tensor(batch_state, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, dtype=torch.int64).view(self.batch_size, self.seq_len, -1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).view(self.batch_size, self.seq_len)

        #get set of next states for end of each sequence
        # fix is needed, batch_next_state torch.Size([500, 6]) but should be the same as batch_state torch.Size([50, 10, 6])
        # batch_next_state1 = list([batch_next_state[i] for i in range(len(batch_next_state)) if (i+1)%(self.seq_len)==0])
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).view(shape)

        # print("returning from prep minibatch")
        # print("batch_state", torch.tensor(batch_state).shape)   # [50, 10, 6]
        # print("batch_action", torch.tensor(batch_action).shape) # [50, 10, 1]
        # print("batch_reward", torch.tensor(batch_reward).shape) # [50, 10]
        # print("batch_next_state", torch.tensor(batch_next_state).shape) # [50, 10, 6]
        # print("batch_next_state1", torch.tensor(batch_next_state1).shape)
        return batch_state, batch_action, batch_reward, batch_next_state

    def train(self):
        total_losses = []
        curr_state = self.env.reset()
        # print("action space: ", self.env.action_space)
        # print("observation space: ", self.env.observation_space)

        epoch_reward = 0

        for i in range(1, 10000):
            action = self.get_action(curr_state)
            prev_state=curr_state
            curr_state, reward, done = self.env.step(action)
            # begin update

            # append tuple to self.nstep_buffer
            # then calculate an expected R which is the weighted sum of all single step
            # rewards, call it R
            # and then transfer the first element in nstep_buffer to self.memory with R and s_
            self.append_to_replay(prev_state, action, reward, curr_state)
            # sample a list of size batch_size from self.memory
            # batch_states: tensor,(BATCH_SIZE, OBS_DIM)    
            # batch_actions: tensor,(BATCH_SIZE, 1)
            # batch_reward: tensor,(BATCH_SIZE, 1)
            if i > self.learn_start:
                batch_states, batch_actions, batch_rewards, batch_next_states = self.prep_minibatch()
                # print("!!!!!!!!!!!!!!!!!!")
                y = self.compute_target(torch.tensor(batch_next_states, dtype=torch.float), batch_rewards)
                loss = self.loss(batch_states, batch_actions, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_count+=1
                if self.update_count % self.target_net_update_freq == 0:
                    with torch.no_grad():
                        self.q_target.load_state_dict(self.q.state_dict())
                total_losses.append(loss.item())
                # end update
                epoch_reward += reward
                print("training loss:", loss.item())
        return total_losses


breakout_env = Breakout()
agent = DRQNModel(obs_dim=breakout_env.observation_space, act_dim= breakout_env.action_space, hidden_dim=4, n_layers=1, batch_size=50, seq_len=10,env=breakout_env)
losses = agent.train()

'''
plt.plot(np.arange(len(losses)), losses)
plt.savefig('3a')
'''