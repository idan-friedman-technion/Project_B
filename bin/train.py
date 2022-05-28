import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.data import Data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import plotly
# import tkinter
import random
from collections import deque
import csv
from utils import stock_state
import tqdm
import datetime as dt
import re
import os
# matplotlib.use('TkAgg')
OBSERVATION_DIM = 8
NUM_OF_ACTIONS = 19
MIN_REPLAY_SIZE = 1000
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TRAIN_EPISODES = 2000


class LambdaLR(): # noqa
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


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


        self.optimizer      = torch.optim.RMSprop(self.model.parameters(), lr=lr) # noqa
        # self.criterion_loss = torch.nn.MSELoss()  # noqa
        self.criterion_loss = torch.nn.SmoothL1Loss()  # noqa
        # model.apply(weights)
        self.writer_loss   = SummaryWriter() # noqa
        self.writer_reward = SummaryWriter()

        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(TRAIN_EPISODES, 0, 1000).step) # noqa

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
        current_qs_list    = self.model(torch.from_numpy(current_states).float().to(DEVICE)) # noqa
        future_qs_list     = self.target_model(torch.from_numpy(new_current_states).float().to(DEVICE)) # noqa

        loss_array = np.array([]) # noqa
        actions    = torch.from_numpy(actions).view(-1, 1).to(DEVICE) # noqa

        # X = current_qs_list.gather(0, actions) # noqa
        Y = torch.from_numpy(rewards).to(DEVICE).float() + discount_factor * future_qs_list.max(dim=1)[0] # noqa

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
        temp = torch.zeros(batch_size).to(DEVICE)
        for i in range(batch_size):
            temp[i] = current_qs_list[i][actions[i]]

        loss = self.criterion_loss(temp, Y)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        a = 1 # noqa
        self.optimizer.step()

        #     tensorboard
        self.writer_loss.add_scalar("Loss/train", loss, curr_episode)

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
            predicted = self.model(torch.from_numpy(observation).float().to(DEVICE)).flatten()
            action = int(torch.argmax(predicted).cpu())
        return action


def read_summary_file(file_path: str = ""):
    with open(file_path, 'r') as f:
        txt = f.read()
        epochs = re.findall(r'After n steps = (\d+)', txt)
        ratios = re.findall(r'final / first reward = (\d+.\d+)', txt)
        loss   = re.findall(r'mean loss = (\d+.\d+)', txt) # noqa
        return np.array(epochs, dtype=int), np.array(ratios, dtype=float), 1000*np.array(loss, dtype=float)


def plot_summary_files(files_list):
    if type(files_list) == str and os.path.isdir(files_list):
        files_names = os.listdir(files_list)
        files_list  = [os.path.join(files_list, fname) for fname in files_names] # noqa
    fig, axs = plt.subplots(2)
    for file in files_list:
        epochs, ratios, loss = read_summary_file(file_path=file)
        axs[0].plot(epochs, loss, linewidth=0.3, label=file)
        axs[1].plot(epochs, ratios, linewidth=0.3, label=file)
        # print(f"{file} - mean last 200 ratio = {np.mean(ratios[-100:])}")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="lower left")
    axs[0].set_title('loss')
    axs[1].set_title('ratios')

    axs[1].axhline(1.62)
    plt.savefig('example.png')
    # plt.show()



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
    for i_out in range(1):
        target_ratio = 1
        # target_ratio = np.random.uniform(1.3, 1.8)

        f = open(f"{str(dt.date.today())}.txt", "w")
        f.write('with mat multiplication, ratio by amount, L1 Loss, RMSprop loss, no TMF-UPRO actions, check if ratio helps\n') # noqa
        epsilon     = 1      # noqa - Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
        max_epsilon = 1      # noqa - You can't explore more than 100% of the time
        min_epsilon = 0.01   # noqa - At a minimum, we'll always explore 1% of the time
        decay       = 0.005   # noqa
        lr          = 0.01  # noqa - learning rate
        change_perc = 0.05

        # An episode a full game
        train_episodes = TRAIN_EPISODES

        # Data
        UPRO_values = read_csv_files('../utils/SPY.csv') # noqa
        TMF_values = read_csv_files('../utils/TYX.csv')   # noqa

        # SPY_values = read_csv_files('../utils/SPY.csv')
        # TYX_values = read_csv_files('../utils/TYX.csv')

        # 1. Initialize the Target and Main models
        # Main Model (updated every 4 steps)
        env_agent = agent(state_shape=OBSERVATION_DIM, action_shape=NUM_OF_ACTIONS, lr=lr)  # noqa - state_space_shape = 3x2 + 1(ration dis) , action_space_shape=3*3*2 + 1

        replay_memory = deque(maxlen=50000)
        target_update_counter = 0

        steps_to_update_target_model = 0

        for episode in tqdm.tqdm(range(train_episodes)):
            # environment
            env = stock_state.env(init_UPRO_val=UPRO_values[0], init_TMF_val=TMF_values[0], num_of_actions=NUM_OF_ACTIONS-1)
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

            reward = env.reward(final_iteration=True)
            if mean_loss is not None: # noqa
                f.write(env.print_env())
                f.write(f'After n steps = {episode}: final / first reward = {float(reward) / float(first_reward) :.2f}\n')
                f.write(f'mean loss = {mean_loss / 1e3 :.4f}*1e3    epsilon = {epsilon:.4f}\n\n\n')
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

            #     tensorboard
            env_agent.writer_reward.add_scalar("reward ratio", float(reward) / float(first_reward), episode)
            # env_agent.lr_scheduler.step()

        f.close()
        env_agent.writer_loss.flush()
        env_agent.writer_loss.close()
        env_agent.writer_reward.flush()
        env_agent.writer_reward.close()


if __name__ == '__main__':
    # print(DEVICE)
    main()
    # plot_summary_files(files_list=['results_files/reference.txt', '2022-05-22.txt'])
    # plot_summary_files(files_list=['target_ratio/2022-05-22-target_1.2068147344802447.txt', 'target_ratio/2022-05-22-target_0.7232387061364003.txt'])
    # file = 'target_ratio/2022-05-22-target_1.2068147344802447.txt'
    # file = 'target_ratio/2022-05-22-target_1.1376639495304905.txt'
    # file = 'target_ratio/2022-05-22-target_1.21685955851548.txt'
    # file = 'target_ratio/2022-05-22-target_0.7232387061364003.txt'

    file = '2022-05-28.txt'
    file1 = '2022-05-25-use_ratio_1.txt'
    plot_summary_files(files_list=[file])