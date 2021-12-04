import numpy as np
import tensorflow as tf
from breakout import Breakout
import tensorflow_probability as tfp
# from policy_gradient import model
# from policy_gradient import AgentReinforce


from tensorflow import keras

# model = keras.models.load_model('policy_gradient_reinforce_model')

breakout_env = Breakout()
breakout_env.make()

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
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
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
        # sum_reward = 0
        # discnt_rewards = []
        # rewards.reverse()
        # for r in rewards:
        #     sum_reward = r + self.gamma*sum_reward
        #     discnt_rewards.append(sum_reward)
        # discnt_rewards.reverse()

        # all_states = tf.convert_to_tensor(reinforce_agent.states)
        # all_actions = tf.convert_to_tensor(reinforce_agent.actions)
        # all_rewards = tf.convert_to_tensor(reinforce_agent.rewards)
        # all_rewards = tf.reshape(all_rewards, [-1])

        with tf.GradientTape() as tape:

            logits = self.model(np.array(states))
            og_logits = logits
            logits = tf.transpose(logits)
            # all_actions = tf.reshape(all_actions, [-1])
            negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            # policy = tf.nn.softmax(logits)
            # log_policy = tf.nn.log_softmax(logits)
            #
            # J = -tf.reduce_mean(negative_likelihoods * sum(rewards))
            #
            # entropy = -tf.reduce_mean(policy*log_policy)

            # logits = self.model(np.array(states), training=True)
            # og_logits = logits
            # logits = tf.transpose(logits)
            print("probabilty of an action sample = ", og_logits[1])
            # print("negative_likelihoods ", negative_likelihoods, "all rewards shape = ", all_rewards.shape)

            # negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            # print("negative_likelihoods shape = ", negative_likelihoods.shape)

            # loss = -J - 0.1*entropy
            loss = -1*tf.reduce_mean(negative_likelihoods * sum(rewards))

            print("loss: ", loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # print("gradient = ", gradients)
        # print("gradients shape = ", len(gradients))
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        # update = tf.train.AdamOptimizer().minimize(loss,var_list=self.model.trainable_variables)

reinforce_agent = AgentReinforce(breakout_env.observation_space, breakout_env.action_space)
reinforce_agent.model.load_weights("weights/policy_gradient_reinforce_model_weights")
# reinforce_agent.model = keras.models.load_model('policy_gradient_reinforce_model')

# model = keras.models.load_model('policy_gradient_reinforce_model')

initial_state = breakout_env.reset()
state = initial_state

# print(reinforce_agent.model(np.array([state])))
# action = reinforce_agent.act(state)
# next_state, reward, done = breakout_env.step(action)

while(True):
    print(state)
    action = reinforce_agent.act(state)
    next_state, reward, done = breakout_env.step(action)
    # next_state, reward, done = breakout_env.step(np.random.randint(3))
    state = next_state

    breakout_env.render()
