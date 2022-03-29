import torch
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.data import Data
import random
import numpy as np
from collections import deque
import csv
from utils import stock_state


class Linear_Model(nn.Module):
    def __init__(self, state_shape=6, action_shape=16):
        super(Linear_Model, self).__init__()
        Linear = [nn.Linear(in_features=state_shape, out_features=action_shape)]
        self.Linear = nn.Sequential(*Linear)
    def forward(self, x):
        return self.Linear(x)


def agent(state_shape=6, action_shape=19):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    # weights = torch.empty(state_shape, action_shape)
    # torch.nn.init.uniform(weights)
    model = Linear_Model(state_shape=state_shape, action_shape=action_shape)
    # model.apply(weights)
    return model

def read_csv_files(file_name):
    stock_val = []
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(spamreader):
            if i == 0:
                continue
            Date, Open, High, Low, Close, Adj_Close, Volume = row
            stock_val.append(np.float(Close))
    return stock_val

def decide_action(observation, model, epsilon):
    random_number = np.random.rand()
    # 2. Explore using the Epsilon Greedy Exploration Strategy
    if random_number <= epsilon:
        # Explore
        action = np.random.randint(0, 19)
    else:
        # Exploit best known action
        # model dims are (batch, env.observation_space.n)
        encoded = observation
        encoded_reshaped = encoded.reshape([-1,])
        predicted = model.predict(encoded_reshaped).flatten()
        action = np.argmax(predicted)
    return action


def train(replay_memory, model, target_model): # model=Q, target_model = Q_tilda
    learning_rate = 0.7 # Learning rate
    discount_factor = 1

    # MIN_REPLAY_SIZE = 1000
    # if len(replay_memory) < MIN_REPLAY_SIZE:
    #     return

    batch_size = np.min([64, len(replay_memory)])
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0].reshape([-1, ]) for transition in mini_batch])
    current_qs_list = model(torch.from_numpy(current_states).float().cuda())
    new_current_states = np.array([transition[3].reshape([-1,]) for transition in mini_batch])
    future_qs_list = target_model(torch.from_numpy(new_current_states).float().cuda())

    X = []
    Y = []

    lr = 0.001
    optimizer     = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion_MSE = torch.nn.MSELoss()

    for index, (observation, action, reward, new_observation) in enumerate(mini_batch):
        max_future_q = reward + discount_factor * torch.max(future_qs_list[index])
        current_qs = current_qs_list[index]
        # current_qs = model(torch.from_numpy(observation.reshape([-1, ])).float())
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)

        loss = criterion_MSE(max_future_q, current_qs[action])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


    # model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
    return model


def main():
    f = open("summary.txt", "w")
    epsilon     = 1      # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1      # You can't explore more than 100% of the time
    min_epsilon = 0.01   # At a minimum, we'll always explore 1% of the time
    decay       = 0.01
    lr          = 0.001  # learning rate
    change_perc = 0.05

    # An episode a full game
    train_episodes = 300

    # Data
    UPRO_values = read_csv_files('../utils/UPRO.csv')
    TMF_values = read_csv_files('../utils/TMF.csv')

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent(state_shape=6, action_shape=19)  # state_space_shape = 3x2 , action_space_shape=3*3*2 + 1
    # Target Model (updated every 100 steps)
    target_model = agent(state_shape=6, action_shape=19)
    target_model.load_state_dict(model.state_dict())

    if torch.cuda.is_available():
        model.cuda()
        target_model.cuda()

    replay_memory = deque(maxlen=512)
    target_update_counter = 0

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        # environment
        env = stock_state.env()
        total_training_rewards = 0
        for i, (upro, tmf) in enumerate(zip(UPRO_values[1:], TMF_values[1:])):
            is_in_ratio = env.is_in_ratio()

            if (np.mod(i,7) != 0) and (1-change_perc < upro/UPRO_values[i] < 1+change_perc) and (1-change_perc < tmf/TMF_values[i] < 1+change_perc) and is_in_ratio:
                continue

            observation = env.observation()
            env.update_stock_values(UPRO_val=upro, TMF_val=tmf)
            steps_to_update_target_model += 1

            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if is_in_ratio:
                action = decide_action(observation=observation, model=model, epsilon=epsilon)
            else:
                action = env.choose_action_for_ratio()
            new_observation, reward = env.step(action)
            replay_memory.append([observation, action, reward, new_observation])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model >= 20:
                model = train(replay_memory, model, target_model)

            total_training_rewards += reward

            if steps_to_update_target_model >= 100:
                print('Copying main network weights to the target network weights')
                target_model.load_state_dict(model.state_dict())
                steps_to_update_target_model = 0

        f.write(f'Total training rewards: {total_training_rewards} after n steps = {episode} with final reward = {reward}')
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    f.close()

if __name__ == '__main__':
    main()