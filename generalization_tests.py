import numpy as np
from breakout import Breakout
import matplotlib.pyplot as plt
from tabular_double_Q_learning import DoubleQLearning
from tabular_Q_learning import QLearning
from Sarsa import Sarsa
import pickle


def BrickGeneralizationTests():

    #QL
    breakout_env_ql = Breakout()
    breakout_env_ql.make()

    breakout_env_ql.reset()

    ballspeeds = list(breakout_env_ql.speeds.values())
    qLearningAgent = QLearning(ballspeeds, decay=-0.0001, steps=30000)

    with open('t_q_learning_saved_policy_values/saved_q_learning_policy.pkl', 'rb') as f:
        qLearningAgent.policy = pickle.load(f)

    with open('t_q_learning_saved_policy_values/saved_q_learning_q_values.pkl', 'rb') as f:
        qLearningAgent.q_values = pickle.load(f)

    # SARSA
    breakout_env_sarsa = Breakout()
    breakout_env_sarsa.make()

    breakout_env_sarsa.reset()

    ballspeeds = list(breakout_env_sarsa.speeds.values())
    sarsaAgent = Sarsa(ballspeeds, decay=-0.0001, steps=30000)

    with open('sarsa_saved_policy_and_q/saved_sarsa_policy.pkl', 'rb') as f:
        sarsaAgent.policy = pickle.load(f)

    with open('sarsa_saved_policy_and_q/saved_sarsa_q_values.pkl', 'rb') as f:
        sarsaAgent.q_values = pickle.load(f)

    # DQL
    breakout_env_dql = Breakout()
    breakout_env_dql.make()

    breakout_env_dql.reset()

    ballspeeds = list(breakout_env_dql.speeds.values())
    doubleQLearningAgent = DoubleQLearning(ballspeeds, decay=-0.0001, steps=30000)

    with open('t_d_q_learning_saved_policy_values/saved_double_q_learning_policy.pkl', 'rb') as f:
        doubleQLearningAgent.policy = pickle.load(f)

    with open('t_d_q_learning_saved_policy_values/saved_double_q_learning_q_values1.pkl', 'rb') as f:
        doubleQLearningAgent.q_values1 = pickle.load(f)

    with open('t_d_q_learning_saved_policy_values/saved_double_q_learning_q_values2.pkl', 'rb') as f:
        doubleQLearningAgent.q_values2 = pickle.load(f)

    # Best trained algorithm: DQL for 100000 episodes
    breakout_env_dql_best = Breakout()
    breakout_env_dql_best.make()

    breakout_env_dql_best.reset()

    ballspeeds = list(breakout_env_dql.speeds.values())
    doubleQLearningAgentBest = DoubleQLearning(ballspeeds, decay=-0.0001, steps=30000)

    with open('t_d_q_learning_saved_policy_values/best/saved_double_q_learning_policy.pkl', 'rb') as f:
        doubleQLearningAgentBest.policy = pickle.load(f)

    with open('t_d_q_learning_saved_policy_values/best/saved_double_q_learning_q_values1.pkl', 'rb') as f:
        doubleQLearningAgentBest.q_values1 = pickle.load(f)

    with open('t_d_q_learning_saved_policy_values/best/saved_double_q_learning_q_values2.pkl', 'rb') as f:
        doubleQLearningAgentBest.q_values2 = pickle.load(f)


    NUMBER_OF_BRICKLAYOUTS = 8
    episodes = 10
    steps = 30000

    scores = []

    for layout in range(NUMBER_OF_BRICKLAYOUTS):
        print("layout = ", layout)

        BestScore_QL = 0
        BestScore_DQL = 0
        BestScore_Best_DQL = 0
        BestScore_SARSA = 0

        breakout_env_dql_best.brickLayout = layout
        breakout_env_dql.brickLayout = layout
        breakout_env_sarsa.brickLayout = layout
        breakout_env_ql.brickLayout = layout

        for episode in range(episodes):
            print("\tepisode = ", episode)
            best_dql_state = tuple(doubleQLearningAgentBest.discretizeStateSpace(breakout_env_dql_best.reset()))
            dql_state = tuple(doubleQLearningAgent.discretizeStateSpace(breakout_env_dql.reset()))
            sarsa_state = tuple(sarsaAgent.discretizeStateSpace(breakout_env_sarsa.reset()))
            ql_state = tuple(qLearningAgent.discretizeStateSpace(breakout_env_ql.reset()))

            for step in range(steps):

                action_ql = qLearningAgent.policy[ql_state]
                next_state_ql, reward_ql, done_ql = breakout_env_ql.step(action_ql)
                ql_state = tuple(qLearningAgent.discretizeStateSpace(next_state_ql))
                BestScore_QL = max(BestScore_QL, breakout_env_ql.score)

                action_dql = doubleQLearningAgent.policy[dql_state]
                next_state_dql, reward_dql, done_dql = breakout_env_dql.step(action_dql)
                dql_state = tuple(doubleQLearningAgent.discretizeStateSpace(next_state_dql))
                BestScore_DQL = max(BestScore_DQL, breakout_env_dql.score)

                action_sarsa = sarsaAgent.policy[sarsa_state]
                next_state_sarsa, reward_sarsa, done_sarsa = breakout_env_sarsa.step(action_sarsa)
                sarsa_state = tuple(sarsaAgent.discretizeStateSpace(next_state_sarsa))
                BestScore_SARSA = max(BestScore_SARSA, breakout_env_sarsa.score)

                action_best_dql = doubleQLearningAgentBest.policy[best_dql_state]
                next_state_best_dql, reward_best_dql, done_best_dql = breakout_env_dql_best.step(action_best_dql)
                best_dql_state = tuple(doubleQLearningAgentBest.discretizeStateSpace(next_state_best_dql))
                BestScore_Best_DQL = max(BestScore_Best_DQL, breakout_env_dql_best.score)

        # print(breakout_env_sarsa.maxScore, breakout_env_ql.maxScore, breakout_env_dql_best.maxScore)
        max_possible_score_for_layout = breakout_env_dql_best.maxScore
        scores.append((BestScore_SARSA, BestScore_QL, BestScore_DQL, BestScore_Best_DQL, max_possible_score_for_layout))

    s_S = []
    s_QL = []
    s_DQL = []
    s_b_DQL = []
    s_max_possible = []

    for layout in range(len(scores)):

        score_SARSA = scores[layout][0]
        s_S.append(score_SARSA)

        score_QL = scores[layout][1]
        s_QL.append(score_QL)

        score_DQL = scores[layout][2]
        s_DQL.append(score_DQL)

        score_Best_DQL = scores[layout][3]
        s_b_DQL.append(score_Best_DQL)

        score_max = scores[layout][4]
        s_max_possible.append(score_max)

        print("Scores for brickLayout: "+ str(layout))
        print("\tBest score achieved by SARSA agent = " + str(score_SARSA))
        print("\tBest score achieved by QL agent = " + str(score_QL))
        print("\tBest score achieved by Double QL agent = " + str(score_DQL))
        print("\tBest score achieved by Best Double QL agent = " + str(score_Best_DQL))
        print("\tBest score possible = " + str(score_max))


    plt.plot(s_DQL, "-b", label="double Q-Learning Score")
    plt.plot(s_QL,"-r", label="Q-Learning Score")
    plt.plot(s_b_DQL,"-k", label="best double Q-Learning Score")
    plt.plot(s_S, "-g", label="SARSA Score")
    plt.plot(s_max_possible, "-m", label="max score possible")
    plt.legend(loc="upper left")
    plt.ylabel('Score')
    plt.xlabel('Brick Layouts')
    plt.title("Best Score acheived by agents in 10 episodes and 30000 steps for different brick layouts")
    plt.savefig('BrickLayouts_generalization_tests.png', bbox_inches='tight')

BrickGeneralizationTests()
