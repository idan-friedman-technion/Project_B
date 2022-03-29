import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import random
import numpy as np
from collections import deque

# An episode a full game
train_episodes = 300
test_episodes = 100

class Linear_Model(nn.Module):
    def __init__(self, state_shape=6, action_shape=16):
        super(Linear_Model, self).__init__()
        self.Linear = nn.Linear(in_features=state_shape, out_features=action_shape)

    def forward(self, x):
        return self.Linear(x)


def agent(state_shape=6, action_shape=19):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    weights = torch.empty(state_shape, action_shape)
    torch.nn.init.uniform(weights)
    model = Linear_Model(state_shape=state_shape, action_shape=action_shape)
    model.apply(weights)



def find_best_action(state):
    a_tag_candidates = []
    for node in state:
        for edge in node:
            a_tag_candidates.append(step(state[node, edge], quantity)) # have to imeplement this function and decide about the quantity
    return max(a_tag_candidates)


def train(replay_memory, model, target_model, done): # model=Q, target_model = Q_tilda
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    # MIN_REPLAY_SIZE = 1000
    # if len(replay_memory) < MIN_REPLAY_SIZE:
    #     return

    batch_size = 64
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def main():
    epsilon     = 1      # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1      # You can't explore more than 100% of the time
    min_epsilon = 0.01   # At a minimum, we'll always explore 1% of the time
    decay       = 0.01
    lr          = 0.001  # learning rate

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent(state_shape=6, action_shape=19)  # state_space_shape = 3x2 , action_space_shape=3*3*2 + 1
    # Target Model (updated every 100 steps)
    taget_model = agent(state_shape=6, action_shape=19)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            if True:
                env.render()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)



