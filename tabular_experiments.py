import numpy as np
from breakout import Breakout
import matplotlib.pyplot as plt
from tabular_double_Q_learning import DoubleQLearning
from tabular_Q_learning import QLearning
from Sarsa import Sarsa
import pickle


breakout_env_ql = Breakout()
breakout_env_1l.make()

breakout_env_ql.reset()

ballspeeds = list(breakout_env_ql.speeds.values())
qLearningAgent = QLearning(ballspeeds, decay=-0.0001, steps=1000)


breakout_env_sarsa = Breakout()
breakout_env_sarsa.make()

breakout_env_sarsa.reset()

ballspeeds = list(breakout_env_sarsa.speeds.values())
sarsaAgent = Sarsa(ballspeeds, decay=-0.0001, steps=1000)

breakout_env_dql = Breakout()
breakout_env_dql.make()

breakout_env_dql.reset()

ballspeeds = list(breakout_env_dql.speeds.values())
doubleQLearningAgent = DoubleQLearning(ballspeeds, decay=-0.0001, steps=1000)

episodes = 10000
rewards_ql_ = []
rewards_dql_ = []
rewards_sarsa_ = []


for e in range(episodes):
    breakout_env_ql.reset()
    breakout_env_dql.reset()
    breakout_env_sarsa.reset()

    states_dql, actions_dql, rewards_dql = doubleQLearningAgent.collect_data(breakout_env_dql)
    doubleQLearningAgent.double_q_learning(states_dql, actions_dql, rewards_dql)
    doubleQLearningAgent.epsilon = doubleQLearningAgent.epsilon + doubleQLearningAgent.decay
    doubleQLearningAgent.k+=1
    if doubleQLearningAgent.epsilon < 0:
        doubleQLearningAgent.epsilon = 0
    rewards_dql_.append(sum(rewards_dql))

    states_ql, actions_ql, rewards_ql = qLearningAgent.collect_data(breakout_env_ql)
    qLearningAgent.q_learning(states_ql, actions_ql, rewards_ql)
    qLearningAgent.epsilon = qLearningAgent.epsilon + qLearningAgent.decay
    qLearningAgent.k+=1
    if qLearningAgent.epsilon < 0:
        qLearningAgent.epsilon = 0

    rewards_ql_.append(sum(rewards_ql))

    states_s, actions_s, rewards_s = sarsaAgent.collect_data(breakout_env_sarsa)
    sarsaAgent.sarsa_learning(states_s, actions_s, rewards_s)
    sarsaAgent.epsilon = sarsaAgent.epsilon + sarsaAgent.decay
    if sarsaAgent.epsilon < 0:
        sarsaAgent.epsilon = 0
    sarsaAgent.k+=1

    rewards_sarsa_.append(sum(rewards_s))

    print("episode = "+str(e)+"epsilon = "+sarsaAgent.epsilon)

plt.plot(rewards_dql_, label="double Q-Learning")
# plt.plot(rewards_ql_, label="Q-Learning")
# plt.plot(rewards_sarsa_, label="SARSA")
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.title("rewards for tabular Algorithms "+str(episodes) + ", epsilon decay = 0.0001, steps = 1000")
plt.savefig('experiment_large_states.png', bbox_inches='tight')
