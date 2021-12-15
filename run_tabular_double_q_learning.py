from breakout import Breakout
from tabular_double_Q_learning import DoubleQLearning
import pickle


breakout_env = Breakout()
breakout_env.make()

ballspeeds = list(breakout_env.speeds.values())
doubleQLearningAgent = DoubleQLearning(ballspeeds)

with open('t_d_q_learning_saved_policy_values/saved_double_q_learning_policy.pkl', 'rb') as f:
    doubleQLearningAgent.policy = pickle.load(f)

with open('t_d_q_learning_saved_policy_values/saved_double_q_learning_q_values1.pkl', 'rb') as f:
    doubleQLearningAgent.q_values1 = pickle.load(f)

with open('t_d_q_learning_saved_policy_values/saved_double_q_learning_q_values2.pkl', 'rb') as f:
    doubleQLearningAgent.q_values2 = pickle.load(f)


initial_state = tuple(doubleQLearningAgent.discretizeStateSpace(breakout_env.reset()))
state = initial_state

i = 1
while(True):
    print("step = ",i, ":: state = ",state, " score = ", breakout_env.score, " max # bricks = ", breakout_env.maxScore)
    action = doubleQLearningAgent.policy[state]
    next_state, reward, done = breakout_env.step(action)
    # next_state, reward, done = breakout_env.step(np.random.randint(3))
    state = tuple(doubleQLearningAgent.discretizeStateSpace(next_state))

    breakout_env.render()
    i+=1
