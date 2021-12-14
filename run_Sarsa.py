from breakout import Breakout
from Sarsa import Sarsa
import pickle


breakout_env = Breakout()
breakout_env.make()

ballspeeds = list(breakout_env.speeds.values())
sarsaAgent = Sarsa(ballspeeds)

with open('sarsa_saved_policy_and_q/saved_sarsa_policy.pkl', 'rb') as f:
    sarsaAgent.policy = pickle.load(f)

with open('sarsa_saved_policy_and_q/saved_sarsa_q_values.pkl', 'rb') as f:
    sarsaAgent.q_values = pickle.load(f)

print("huh?")


initial_state = tuple(sarsaAgent.discretizeStateSpace(breakout_env.reset()))
state = initial_state

i = 1
while(True):
    print("step = ",i, ":: state = ",state)
    action = sarsaAgent.policy[state]
    next_state, reward, done = breakout_env.step(action)
    # next_state, reward, done = breakout_env.step(np.random.randint(3))
    state = tuple(sarsaAgent.discretizeStateSpace(next_state))

    breakout_env.render()
    i+=1
