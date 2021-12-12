from breakout import Breakout
from tabular_Q_learning import QLearning
import pickle


breakout_env = Breakout()
breakout_env.make()

ballspeeds = list(breakout_env.speeds.values())
qLearningAgent = QLearning(ballspeeds)

with open('t_q_learning_saved_policy_values/saved_q_learning_policy.pkl', 'rb') as f:
    qLearningAgent.policy = pickle.load(f)

with open('t_q_learning_saved_policy_values/saved_q_learning_q_values.pkl', 'rb') as f:
    qLearningAgent.q_values = pickle.load(f)

# print("huh?")


initial_state = tuple(qLearningAgent.discretizeStateSpace(breakout_env.reset()))
state = initial_state

i = 1
while(True):
    print("step = ",i, ":: state = ",state)
    action = qLearningAgent.policy[state]
    next_state, reward, done = breakout_env.step(action)
    # next_state, reward, done = breakout_env.step(np.random.randint(3))
    state = tuple(qLearningAgent.discretizeStateSpace(next_state))

    breakout_env.render()
    i+=1
