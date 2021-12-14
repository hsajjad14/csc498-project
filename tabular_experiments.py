import numpy as np
from breakout import Breakout
import matplotlib.pyplot as plt
from tabular_double_Q_learning import DoubleQLearning
from tabular_Q_learning import QLearning
from Sarsa import Sarsa
import pickle


def experiment1():

    breakout_env_ql = Breakout()
    breakout_env_ql.make()

    breakout_env_ql.reset()

    ballspeeds = list(breakout_env_ql.speeds.values())
    qLearningAgent = QLearning(ballspeeds, decay=-0.0001, steps=5000)

    # print("here 1 ")


    breakout_env_sarsa = Breakout()
    breakout_env_sarsa.make()

    breakout_env_sarsa.reset()

    ballspeeds = list(breakout_env_sarsa.speeds.values())
    sarsaAgent = Sarsa(ballspeeds, decay=-0.0001, steps=5000)

    # print("here 2 ")


    breakout_env_dql = Breakout()
    breakout_env_dql.make()

    breakout_env_dql.reset()

    ballspeeds = list(breakout_env_dql.speeds.values())
    doubleQLearningAgent = DoubleQLearning(ballspeeds, decay=-0.0001, steps=5000)

    # print("here 3 ")


    episodes = 10000
    rewards_ql_ = []
    rewards_dql_ = []
    rewards_sarsa_ = []


    for e in range(episodes):
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

        print("episode = "+str(e)+", epsilon = "+str(doubleQLearningAgent.epsilon))

    plt.plot(rewards_dql_, "-b", label="double Q-Learning")
    plt.plot(rewards_ql_,"-r", label="Q-Learning")
    plt.plot(rewards_sarsa_, "-g", label="SARSA")
    plt.legend(loc="upper left")
    plt.ylabel('rewards')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.title("QL_and_DQL Rewards for tabular Algorithms "+str(episodes) + ", epsilon decay = 0.0001, steps = "+str(doubleQLearningAgent.steps))
    plt.savefig('QL_and_DQL_Rewards_Comparison.png', bbox_inches='tight')

    plt.clf()

    plt.plot(rewards_dql_[-2000:], "-b", label="double Q-Learning")
    plt.plot(rewards_ql_[-2000:],"-r", label="Q-Learning")
    plt.plot(rewards_sarsa_[-2000:], "-g", label="SARSA")
    plt.legend(loc="upper left")
    plt.ylabel('rewards')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.title("Last 2000 episodes; Rewards for QL_and_DQL Algorithms "+ ", epsilon decay = 0.0001, steps = "+str(doubleQLearningAgent.steps))
    plt.savefig('Last_2000_rewards_QL_and_DQL_comparison.png', bbox_inches='tight')


    with open('saved_double_q_learning_policy.pkl', 'wb') as f:
        pickle.dump(doubleQLearningAgent.policy, f)

    with open('saved_double_q_learning_q_values1.pkl', 'wb') as f:
        pickle.dump(doubleQLearningAgent.q_values1, f)

    with open('saved_double_q_learning_q_values2.pkl', 'wb') as f:
        pickle.dump(doubleQLearningAgent.q_values2, f)


    with open('saved_q_learning_policy.pkl', 'wb') as f:
        pickle.dump(qLearningAgent.policy, f)

    with open('saved_q_learning_q_values.pkl', 'wb') as f:
        pickle.dump(qLearningAgent.q_values, f)

    with open('saved_sarsa_policy.pkl', 'wb') as f:
        pickle.dump(sarsaAgent.policy, f)

    with open('saved_sarsa_q_values.pkl', 'wb') as f:
        pickle.dump(sarsaAgent.q_values, f)

def experiment2():
    trainDoubleQLearning()
    trainQLearning()
    trainSarsa()

def trainSarsa():
    # sarsa
    breakout_env = Breakout()
    breakout_env.make()

    breakout_env.reset()

    ballspeeds = list(breakout_env.speeds.values())
    sarsaAgent = Sarsa(ballspeeds, decay=-0.000026, steps=21000)
    sarsaAgent.train()

def trainQLearning():
    # q-learning
    breakout_env = Breakout()
    breakout_env.make()

    breakout_env.reset()

    ballspeeds = list(breakout_env.speeds.values())
    qLearningAgent = QLearning(ballspeeds, decay=-0.000026, steps=21000)
    qLearningAgent.train()

def trainDoubleQLearning():
    # double q-Learning
    print("training dql")
    breakout_env = Breakout()
    breakout_env.make()

    breakout_env.reset()

    ballspeeds = list(breakout_env.speeds.values())
    doubleQLearningAgent = DoubleQLearning(ballspeeds, decay=-0.000026, steps=21000)
    doubleQLearningAgent.train()

# experiment2()
