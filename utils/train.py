import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data
import random
import numpy as np

# An episode a full game
train_episodes = 300
test_episodes = 100


def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    weights = torch.empty(state_shape, action_shape)
    torch.nn.init.uniform(weights)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 20, 5),
        torch.nn.ReLU(),
        torch.nn.Conv2d(20, 64, 5),
        torch.nn.ReLU()
    )
    return model

def get_state_mat(line_index, cur_UPRO_hold, cur_TMF_hold, cur_money_hold, cur_wieghts): #cur weights should be in size: 3X2
    "creates the matirx which represent the state"
    A = torch.tensor([[0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0]]) # 3X2
    f_UPRO = [UPROExcel[line_index][5], cur_UPRO_hold]
    f_TMF = [TMFExcel[line_index][5], cur_TMF_hold]
    f_money = [1, cur_money_hold]
    F = [f_UPRO, f_TMF, f_money] # 2X2
    return np.matmul(A,F)*cur_wieghts #output is 3X2


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




