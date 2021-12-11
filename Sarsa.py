#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from breakout import Breakout
import matplotlib.pyplot as plt
import pickle


# In[2]:


breakout_env = Breakout()
breakout_env.make()


# In[143]:


# ep = 0.9*1000
# for e in range(1,10000):
#     ep = ep/e*200
#     print(e, ep)


# In[154]:


class Sarsa():

    def __init__(self, ballSpeeds, gamma=0.9, epsilon=1):
        self.paddleXLocations, self.ballXLocations, self.ballYLocations =  self.discretizeStateSpaceAllStates(800, 600)

        self.ballSpeeds = ballSpeeds

        # policy estimate for each state
        self.policy = {}

        # Q-value estimate in each state
        self.q_values  = {}

        for px in self.paddleXLocations:
            for bx in self.ballXLocations:
                for by in self.ballYLocations:
                    for ballSpeed in self.ballSpeeds:
                        # default action is to do nothing
                        self.policy[(px, bx, by, ballSpeed[0], ballSpeed[1])] = 2
                        for action in range(3):
                            self.q_values[((px, bx, by, ballSpeed[0], ballSpeed[1]), action)] = 0


        self.alpha = 0.9
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = -0.00002
        self.k = 1


    def sarsa_learning(self, states, actions, rewards):
        # self.epsilon = 0.9*1000

        for t in range(len(states)):
            state = tuple(states[t])
            reward = rewards[t]
            action = actions[t]
            # print(state)
            # print(reward)
            # print(action)
            # break



            if t < len(states) - 1:
                nextState = tuple(states[t+1])
                nextAction = actions[t+1]

                currentQValue = self.q_values[(state, action)]
                nextQValue = self.q_values[(nextState, nextAction)]

                self.q_values[(state, action)] = currentQValue + self.alpha * (reward + self.gamma*nextQValue - currentQValue)

            QAction0 = self.q_values[(state, 0)]
            QAction1 = self.q_values[(state, 1)]
            QAction2 = self.q_values[(state, 2)]

            bestActions = [QAction0, QAction1, QAction2]

            bestAction = max(enumerate(bestActions), key=lambda x: x[1])[0]

            random_probability = np.random.random()
            at = 0

            if random_probability < 1 - self.epsilon:
                # evaluate best action
                self.policy[state] = bestAction
            else:
                # explore random action
                self.policy[state] = np.random.randint(3)


        # self.epsilon = self.epsilon/t*100



    def epsilon_greedy_policy(self, obs):
        """
        obs: integer representing state

        returns integer representing action for current state,
        according to epsilon-greedy policy (see handout)

        epsilon is stored in self.epsilon

        Hint:
        act = random_agent(obs) #obtains a random action for obs
        act = self.__call__(obs) #obtains action according to self.policy
        """

        # get next action
        random_probability = np.random.random()
        at = 0

        if random_probability < 1 - self.epsilon:
            # evaluate best action
            at = self.policy[obs]
        else:
            # explore random action
            at = np.random.randint(3)

        # update epsilon, it goes to 0
        # so we eventually stop exploring and instead start evaluating
#         self.epsilon = self.epsilon/self.k*100

        return at


    # given state [paddle x-location, ball x-location, ball y-location, ball x-speed, ball y-speed, bricks left] from environment
    # descretize state space for each variable and remove bricks left
    # so paddle x-locations are split into 10 possible locations (because paddle is 80 pixels wide and game screen is 800)
    # ball x-locations are split into 40 possible locations, ball y-location into 20
    def discretizeStateSpace(self, state):
        paddleXLocation = state[0]
        ballXLocation = state[1]
        ballYLocation = state[2]
        ballSpeed = [state[3], state[4]]

        stateToReturn = [0,0,0,ballSpeed[0], ballSpeed[1]]

        for px in self.paddleXLocations:
            if paddleXLocation <= px:
                stateToReturn[0] = px
                break

        for bx in self.ballXLocations:
            if ballXLocation <= bx:
                stateToReturn[1] = bx
                break

        for by in self.ballYLocations:
            if ballYLocation <= by:
                stateToReturn[2] = by
                break

        return stateToReturn




    # get all possible paddle x-locations, ball x-locations, and ball y-locations
    # so paddle x-locations are split into 10 possible locations (because paddle is 80 pixels wide and game screen is 800)
    # ball x-locations are split into 40 possible locations, ball y-location into 20
    # def discretizeStateSpaceAllStates(self, screenWidth, screenHeight):
    #     # for paddle x-locations, split screenWidth by 80 (800/80 = 10 locations)
    #     paddleXLocations = []
    #     for i in range(0, screenWidth, 80):
    #         paddleXLocations.append(i)
    #
    #
    #     # for ball x-locations, split screenWidth by 20 (800/20 = 40 locations)
    #     ballXLocations = []
    #     for i in range(0, screenWidth, 20):
    #         ballXLocations.append(i)
    #
    #     # for ball y-locations, split screenWidth by 30 to get 20 locations (600/30 = 20 locations)
    #     ballYLocations = []
    #     for i in range(0, screenHeight, 30):
    #         ballYLocations.append(i)
    #
    #     return paddleXLocations, ballXLocations, ballYLocations

    # get all possible paddle x-locations, ball x-locations, and ball y-locations
    # so paddle x-locations are split into 10 possible locations (because paddle is 80 pixels wide and game screen is 800)
    # ball x-locations are split into 40 possible locations, ball y-location into 20
    # more states
    def discretizeStateSpaceAllStates(self, screenWidth, screenHeight):
        # for paddle x-locations, split screenWidth by 80 (800/80 = 10)
        paddleXLocations = []
        for i in range(0, screenWidth, 80):
            paddleXLocations.append(i)


        # for ball x-locations, split screenWidth by 10 (800/10 = 80 locations)
        ballXLocations = []
        for i in range(0, screenWidth, 10):
            ballXLocations.append(i)

        # for ball y-locations, split screenWidth by 20 to get 30 locations (600/20 = 30 locations)
        ballYLocations = []
        for i in range(0, screenHeight, 20):
            ballYLocations.append(i)

        return paddleXLocations, ballXLocations, ballYLocations


    def collect_data(self, env):
        obs = self.discretizeStateSpace(env.reset())

        rewards = []
        states = []
        actions = []
#         self.k = 0

        # for step in range(15000): # A
        for step in range(8000):
            states.append(obs)
            act = self.epsilon_greedy_policy(tuple(obs))
#             act = self.policy[tuple(obs)]
            obs, rew, done = env.step(act)
            obs = self.discretizeStateSpace(obs)
            rewards.append(rew)
            actions.append(act)
#             self.k+=1

        states[-1] = obs

        return states, actions, rewards




# In[155]:


# max(enumerate([1,2,3]), key=lambda x: x[1])[0]


# In[ ]:

#
# breakout_env.reset()
#
# ballspeeds = list(breakout_env.speeds.values())
# sarsaAgent = Sarsa(ballspeeds)
#
# episodes = 60000
#
# for e in range(episodes):
#     data = sarsaAgent.collect_data(breakout_env)
#     sarsaAgent.sarsa_learning(*data)
#     sarsaAgent.epsilon = sarsaAgent.epsilon + sarsaAgent.decay
#     sarsaAgent.k+=1
#     print("episode =", e)
#
# with open('saved_policy.pkl', 'wb') as f:
#     pickle.dump(sarsaAgent.policy, f)
#
# with open('saved_q_values.pkl', 'wb') as f:
#     pickle.dump(sarsaAgent.q_values, f)

# print(*data[0])


# In[153]:


# # Final Benchmarking
# obs = tuple(sarsaAgent.discretizeStateSpace(breakout_env.reset()))

# rewards = np.zeros((100, 100))
# states = np.zeros((100, 101))
# actions = np.zeros((100, 100))

# for run in range(100):
#     for step in range(100):
# #         states[run, step] = obs
#         act = sarsaAgent.policy[obs]
#         obs, rew, done = breakout_env.step(act)
#         obs = tuple(sarsaAgent.discretizeStateSpace(obs))
#         rewards[run, step] = rew
#         actions[run, step] = act
# #     states[run, -1] = obs

# print("Average return: {}".format(rewards.sum(1).mean()))
# print("Standard deviation: {}".format(rewards.sum(1).std()))


# In[149]:


# l = sarsaAgent.q_values
# val = []
# for k, v in l.items():
#     if v != 0:
#         val.append(v)


# In[79]:


# ballspeeds = breakout_env.speeds.values()
# list(ballspeeds)


# In[80]:


# ballspeeds = list(breakout_env.speeds.values())
# sarsaAgent = Sarsa(ballspeeds)


# In[81]:


# state = breakout_env.step(0)[0]
# descretedState = sarsaAgent.discretizeStateSpace(state)


# In[82]:


# state


# In[83]:


# descretedState


# In[50]:


# breakout_env.reset()[0]


# In[ ]:

# breakout_env = Breakout()
# breakout_env.make()
#
# initial_state = tuple(sarsaAgent.discretizeStateSpace(breakout_env.reset()))
# state = initial_state
#
# while(True):
#     print(state)
#     action = sarsaAgent.policy[state]
#     next_state, reward, done = breakout_env.step(action)
#     # next_state, reward, done = breakout_env.step(np.random.randint(3))
#     state = tuple(sarsaAgent.discretizeStateSpace(next_state))
#
#     breakout_env.render()
