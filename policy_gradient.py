#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from breakout import Breakout

breakout_env = Breakout()

print("action space: ", breakout_env.action_space)
print("observation space: ", breakout_env.observation_space)

# run this initially to make the brick layout
breakout_env.make()

def roundToNearest(numbers, toRound):
    distance = np.zeros(len(numbers))
    for i in range(len(numbers)):
        distance[i] = abs(numbers[i] - toRound)

    return np.argmin(distance)

# using https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24 instead

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
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0000000000001)
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

    def a_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        # print("log_prob",log_prob, "action: ", action, "dist: ", dist, "prob: ", prob)
        loss = -log_prob*reward
        return loss

    def train(self, states, rewards, actions):

        with tf.GradientTape() as tape:

            logits = self.model(np.array(states))
            og_logits = logits
            logits = tf.transpose(logits)
            # all_actions = tf.reshape(all_actions, [-1])
            negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits)

            print("probabilty of an action sample = ", og_logits[1])

            loss = -tf.reduce_mean(negative_likelihoods * sum(rewards))/1000

            print("loss: ", loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))





breakout_env.reset()
reinforce_agent = AgentReinforce(breakout_env.observation_space, breakout_env.action_space)
episodes = 100000
epsilon = 0.9*1000
largest_reward = -10000000000000000000

for e in range(1, episodes+1):

    done = False
    state = breakout_env.reset()
    total_reward = 0
    rewards = []
    states = []
    actions = []
    for i in range(5000):
        #env.render()
        # alternate
        p = np.random.random()
        if p < 1 - epsilon:
            action = reinforce_agent.act(state)
        else:
            action = np.random.randint(3)
            # print(action)


        if e > 700:
            p2 = np.random.randint(7)
            if p2 == 5:
                action = np.random.randint(3)
            else:
                action = reinforce_agent.act(state)

        # action = reinforce_agent.act(state)
        next_state, reward, done = breakout_env.step(action)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        state = next_state
        total_reward += reward

    if total_reward > largest_reward:
        print("saved weights")
        largest_reward = total_reward
        reinforce_agent.model.save_weights("weights/policy_gradient_reinforce_model_weights")

    # print("----- \t actions = ",actions, "\n")
    reinforce_agent.episodes += 1
    reinforce_agent.states += states
    reinforce_agent.rewards += rewards
    reinforce_agent.actions += actions
    reinforce_agent.train(states, rewards, actions)
    epsilon = (epsilon/e)*300
    #print("total step for this episord are {}".format(t))
    print("total reward after {} steps is {}\n".format(e, total_reward))


# In[ ]:


print("action space: ", breakout_env.action_space)
print("observation space: ", breakout_env.observation_space)
