import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.data import Data
import random
import numpy as np
from collections import deque
import csv
from utils import stock_state
import tqdm
import datetime as dt


NUM_OF_ACTIONS = 19
MIN_REPLAY_SIZE = 1000


class Linear_Model(nn.Module): # noqa
    def __init__(self, state_shape=6, action_shape=19):
        super(Linear_Model, self).__init__()
        self.Linear = nn.Linear(in_features=state_shape, out_features=action_shape) # noqa
        # self.Linear = nn.Sequential(Linear)

    def forward(self, x):
        return self.Linear(x)


class agent(): # noqa
    def __init__(self, state_shape=6, action_shape=19, lr=0.001):
        # weights = torch.empty(state_shape, action_shape)
        # torch.nn.init.uniform(weights)
        self.actions_shape = action_shape
        self.model = Linear_Model(state_shape=state_shape, action_shape=action_shape)
        # Target Model (updated every 100 steps)
        self.target_model = Linear_Model(state_shape=state_shape, action_shape=action_shape)
        self.target_model.load_state_dict(self.model.state_dict())
        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()


        self.optimizer     = torch.optim.Adam(self.model.parameters(), lr=lr) # noqa
        self.criterion_MSE = torch.nn.MSELoss()  # noqa
        # model.apply(weights)
        self.writer = SummaryWriter()

    def copy_models(self):
        self.target_model.load_state_dict(self.model.state_dict())
        return

    def train(self, replay_memory, curr_episode, batch_size=64):  # model=Q, target_model = Q_tilda
        learning_rate   = 0.7  # noqa - Learning rate
        discount_factor = 0.618

        if len(replay_memory) < MIN_REPLAY_SIZE:
            return None

        mini_batch         = random.sample(replay_memory, batch_size) # noqa
        current_states     = np.array([transition[0].reshape([-1, ]) for transition in mini_batch]) # noqa
        actions            = np.array([transition[1]                 for transition in mini_batch]) # noqa
        rewards            = np.array([transition[2]                 for transition in mini_batch]) # noqa
        new_current_states = np.array([transition[3].reshape([-1, ]) for transition in mini_batch]) # noqa
        current_qs_list    = self.model(torch.from_numpy(current_states).float().cuda()) # noqa
        future_qs_list     = self.target_model(torch.from_numpy(new_current_states).float().cuda()) # noqa

        loss_array = np.array([]) # noqa
        actions    = torch.from_numpy(actions).view(-1, 1).cuda() # noqa

        # X = current_qs_list.gather(0, actions) # noqa
        Y = torch.from_numpy(rewards).cuda().float() + discount_factor * future_qs_list.max(dim=1)[0] # noqa

        # for index, (observation, action, reward, new_observation) in enumerate(mini_batch):
        #     max_future_q = reward + discount_factor * torch.max(future_qs_list[index])
        #     current_qs = current_qs_list[index][action]
        #     current_qs = self.model(torch.from_numpy(observation.reshape([-1, ])).float().cuda())[action] # noqa
        #     current_qs = model(torch.from_numpy(observation.reshape([-1, ])).float())
        #     current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q
        #
        #     X.append(max_future_q)
        #     Y.append(current_qs)

        # loss = self.criterion_MSE(max_future_q, current_qs)
        # temp = current_qs_list.gather(0, actions)
        temp = np.zeros(batch_size)
        index = 0
        while index < batch_size:
            temp[index] = current_qs_list[index][actions[index]]
            index += 1
        temp = torch.from_numpy(temp).cuda().float()

        loss = self.criterion_MSE(temp, Y)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        a = 1 # noqa
        self.optimizer.step()

        #     tensorboard
        self.writer.add_scalar("Loss/train", loss, curr_episode)

        loss_array = np.append(loss_array, loss.item())
        return np.mean(loss_array)

    def decide_action(self, observation, epsilon):
        random_number = np.random.rand()
        # 2. Explore using the Epsilon Greedy Exploration Strategy
        if random_number <= epsilon:
            # Explore
            action = np.random.randint(0, self.actions_shape)
        else:
            # Exploit best known action
            # model dims are (batch, env.observation_space.n)
            predicted = self.model(torch.from_numpy(observation).float().cuda()).flatten()
            action = int(torch.argmax(predicted).cpu())
        return action


def read_csv_files(file_name):
    stock_val = []
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',') # noqa
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            Date, Open, High, Low, Close, Adj_Close, Volume = row # noqa
            stock_val.append(np.float(Close))
    return stock_val


def main():
    np.random.seed(30)
    f = open(f"{str(dt.date.today())}-results3.txt", "w")
    f.write('with mat multiplication, ratio added to observation/reward, no tax in reward, tax both direction in selling, ratio by amount only\n') # noqa
    epsilon     = 1      # noqa - Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1      # noqa - You can't explore more than 100% of the time
    min_epsilon = 0.01   # noqa - At a minimum, we'll always explore 1% of the time
    decay       = 0.005   # noqa
    lr          = 0.001  # noqa - learning rate
    change_perc = 0.05

    # An episode a full game
    train_episodes = 2000

    # Data
    UPRO_values = read_csv_files('../utils/UPRO.csv') # noqa
    TMF_values = read_csv_files('../utils/TMF.csv')   # noqa

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    env_agent = agent(state_shape=7, action_shape=NUM_OF_ACTIONS)  # noqa - state_space_shape = 3x2 + 1(ration dis) , action_space_shape=3*3*2 + 1

    replay_memory = deque(maxlen=50000)
    target_update_counter = 0

    steps_to_update_target_model = 0

    for episode in tqdm.tqdm(range(train_episodes)):
        # environment
        env = stock_state.env(num_of_actions=NUM_OF_ACTIONS-1)
        total_training_rewards = 0
        first_reward           = env.reward() # noqa
        for i, (upro, tmf) in enumerate(zip(UPRO_values[1:], TMF_values[1:])): # noqa

            if (np.mod(i, 20) != 0) and (1-change_perc < upro/UPRO_values[i] < 1+change_perc) and (1-change_perc < tmf/TMF_values[i] < 1+change_perc): # noqa
                continue

            steps_to_update_target_model += 1

            observation = env.observation()
            env.update_stock_values(UPRO_val=upro, TMF_val=tmf)

            # 2. Explore using the Epsilon Greedy Exploration Strategy
            action = env_agent.decide_action(observation=observation, epsilon=epsilon)
            new_observation, reward = env.step(action)
            replay_memory.append([observation, action, reward, new_observation])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model >= 20:
                mean_loss = env_agent.train(replay_memory=replay_memory, curr_episode=episode)

            total_training_rewards += reward

            if steps_to_update_target_model >= 100:
                # print('Copying main network weights to the target network weights')
                env_agent.copy_models()
                steps_to_update_target_model = 0


        if mean_loss is not None: # noqa
            f.write(env.print_env())
            f.write(f'After n steps = {episode}: final / first reward = {float(reward) / float(first_reward) :.2f}\n')
            f.write(f'mean loss = {mean_loss / 1e9 :.4f}*1e9    epsilon = epsilon = {epsilon:.4f}\n\n\n')
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    f.close()
    env_agent.writer.flush()
    env_agent.writer.close()


if __name__ == '__main__':
    main()
