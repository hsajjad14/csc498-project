#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from breakout import Breakout


# In[2]:


breakout_env = Breakout()


# In[79]:


print("action space: ", breakout_env.action_space)
print("observation space: ", breakout_env.observation_space)


# In[3]:


# running this will run the playable game
# breakout_env.main()


# In[3]:


# run this initially to make the brick layout
breakout_env.make()


# In[4]:


print(breakout_env.step(1))


# In[5]:


print(breakout_env.step(0))


# In[24]:


print(breakout_env.step(0))


# In[25]:


print(breakout_env.step(0))


# In[26]:


print(breakout_env.step(0))


# In[27]:


print(breakout_env.step(0))


# In[28]:


print(breakout_env.step(0))


# In[11]:


class Agent():
    def __init__(self, observation_dim, params = None, action_bounds = None):
        pass

    def __call__(self, obs):
        return self.act(obs)


# In[12]:


# REINFORCE


# In[13]:


N_EPISODES = 50 # Change this if you find this is not sufficient


# In[14]:


l = np.random.random(10)
print(l)
np.argmin(l)


# In[15]:


def roundToNearest(numbers, toRound):
    distance = np.zeros(len(numbers))
    for i in range(len(numbers)):
        distance[i] = abs(numbers[i] - toRound)

    return np.argmin(distance)



# In[16]:


roundToNearest([1,20, 50, 100], 11)


# In[13]:


class GaussianPolicy(Agent):

    def __init__(self, observation_dim, params = None, action_bounds = None,
                 gamma=0.9, alpha = 0.1, sigma = 0.1):
        self.states = observation_dim # dimension of observations

        # Action bounds are limits of shape (2,),
        # representing the lower/upper limits on the action
        self.action_bounds = action_bounds

        self.gamma = gamma

        # Initial parameters of the policy
        self.params = params
        self.sigma = sigma

        self.alpha = alpha

    def act(self, obs):
        '''
        obs: Array of size (n, observation_dims) representing a batch of observations,
        which are each a vector of (observation_dims)

        Returns:
        actions: Array of size (n, 1) representing actions determined by params, self.sigma
        '''
#         print(obs)
        theta = self.params
        states = obs

        a = np.random.normal(loc=states.dot(theta))

        ret = (1/np.sqrt(2*np.pi*self.sigma))*np.exp((-1/(2*self.sigma)) * (a - states.dot(theta)))
        possible_actions = [0,1,2]
        clipped_action = roundToNearest(possible_actions, ret)

        return clipped_action # TODO: Write code here

    def update(self, obs, actions, rewards):
        '''
        obs: Array of size (n+1, observation_dims) representing a batch of observations,
        which are each a vector of (observation_dims)
        rewards: Array of size (n,) representing a batch of rewards
        actions: Array of size (n,) representing a batch of actions

        Returns:
        Nothing, modifies parameters according to the REINFORCE algorithm
        '''

        cost = 0
        for i in range(len(obs)):
            if i != len(obs) - 1:
                states = obs[i]
                action = actions[i]
                reward = rewards[i]

                gaussian_grad = ((action - states.dot(self.params))*states)/self.sigma

                score = gaussian_grad*reward

                cost+=score

        self.params = self.params + self.alpha*cost

        return # TODO: Write code here

    def collect_data(self, task):
        # Do not modify
        obs = task.reset()

        rewards = np.zeros((100,))
        states = np.zeros((101, self.states))
        actions = np.zeros((100, 1))

        for step in range(100):
            states[step, :] = obs
            act = self.__call__(obs)
#             print(act)
            obs, rew, done = task.step(act)
            rewards[step] = rew
            actions[step] = act
#             print(rew)

        states[-1, :] = obs
#         print(states, "--")
#         print(rewards)

        return states, actions, rewards

# Do not modify
# init_mu = np.random.random((breakout_env.observation_space,))
#
# agent = GaussianPolicy(breakout_env.observation_space, init_mu, [breakout_env.min_action, breakout_env.max_action])
#
# rewards = []
#
# for i in range(N_EPISODES):
#     data = agent.collect_data(breakout_env)
#
#     rewards.append(np.mean(data[2]))
#     agent.update(*data)
#
# plt.plot(np.arange(N_EPISODES), rewards)
# # Final Benchmarking
# rewards = np.zeros((100, 100))
#
# for i in range(100):
#     rewards[i] = agent.collect_data(breakout_env)[2]
#
# print("Average return: {}".format(rewards.sum(1).mean()))
# print("Standard deviation: {}".format(rewards.sum(1).std()))
#

# In[14]:


# using https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24 instead


# In[6]:


class model(tf.keras.Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(30,input_dim=observation_space,activation='relu')
        self.d2 = tf.keras.layers.Dense(30,activation='relu')
        self.out = tf.keras.layers.Dense(action_space,activation='softmax')

    def call(self, input_data):
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

    def action_value(self, state):
        p = self.predict(state)
        return p


# In[7]:


class AgentReinforce():
    def __init__(self, observation_space, action_space):
        self.model = model(observation_space, action_space)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 1
        self.episodes = 1
        self.rewards = []
        self.states = []
        self.actions = []

    def act(self,state):
        prob = self.model(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        # print("probabillity of actions = ", prob)
        if int(action.numpy()[0]) > 2:
            return 2
        elif int(action.numpy()[0]) < 0:
            return 0

        return int(action.numpy()[0])

    def a_loss(self,prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss

    def train(self, states, rewards, actions):
        all_states = tf.convert_to_tensor(reinforce_agent.states)
        all_actions = tf.convert_to_tensor(reinforce_agent.actions)
        all_rewards = tf.convert_to_tensor(reinforce_agent.rewards)
        all_rewards = tf.reshape(all_rewards, [-1])

        logits = self.model(np.array(all_states))
        og_logits = logits
        logits = tf.transpose(logits)
        all_actions = tf.reshape(all_actions, [-1])
        negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=all_actions, logits=logits)
        # print(negative_likelihoods.shape, tf.convert_to_tensor(rewards).shape)

        loss = tf.reduce_mean(negative_likelihoods)
        loss = loss * tf.reduce_sum(all_rewards) / 70000000000
        # loss = loss * sum(rewards) / 7000000
        print("\none state prob dist for actions = ", og_logits[1])
        print("loss in this episode = ", loss)

        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss, self.model.trainable_variables)
            for g in gradients:
                if g:
                    self.opt.apply_gradients(zip(g, self.model.trainable_variables))

# In[8]:


breakout_env.reset()
reinforce_agent = AgentReinforce(breakout_env.observation_space, breakout_env.action_space)
episodes = 10000
# epsilon = 0.9*100
for e in range(1, episodes+1):

    done = False
    state = breakout_env.reset()
    total_reward = 0
    rewards = []
    states = []
    actions = []
    for i in range(1500):
        #env.render()
        action = reinforce_agent.act(state)
        next_state, reward, done = breakout_env.step(action)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        state = next_state
        total_reward += reward

    reinforce_agent.episodes += 1
    reinforce_agent.states += states
    reinforce_agent.rewards += rewards
    reinforce_agent.actions += actions
    reinforce_agent.train(states, rewards, actions)
    # epsilon = (epsilon/e)*100
    #print("total step for this episord are {}".format(t))
    print("total reward after {} steps is {}".format(e, total_reward))


# In[ ]:


print("action space: ", breakout_env.action_space)
print("observation space: ", breakout_env.observation_space)
