# import torch
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from collections import deque
# import csv
# from utils_3 import stock_state
from utils_3.Helper_Functions import *

import tqdm
import re
import datetime
import os
import torch
from torch.utils.data import DataLoader, Dataset

# matplotlib.use('TkAgg')
# Network variables
OBSERVATION_DIM = 7
NUM_OF_ACTIONS = 6
MIN_REPLAY_SIZE = 256
MAX_RB_LENGTH   = 10000 # noqa
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# lr variables
TRAIN_EPISODES = 3000
use_lr_decay = 1
L1_BETA = 20
START_EP = 0
MAX_EPSILON = 1

# weights variables
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "model.pth")
UPLOAD_WEIGHTS = 0

RUN_MAIN = 1
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), f"{str(datetime.date.today())}_final_run.txt")



class agent(): # noqa
    def __init__(self, num_node_features=6, action_shape=NUM_OF_ACTIONS, lr=0.001, gamma=0.9, train_episodes=TRAIN_EPISODES):
        # weights = torch.empty(state_shape, action_shape)
        # torch.nn.init.uniform(weights)
        self.actions_shape = action_shape
        self.model = GCN(num_classes=action_shape)
        self.gamma = gamma

        if UPLOAD_WEIGHTS:
            self.model.load_state_dict(torch.load(WEIGHTS_FILE))

        # Target Model (updated every 100 steps)
        self.target_model = GCN()
        self.target_model.load_state_dict(self.model.state_dict())
        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()


        self.optimizer      = torch.optim.RMSprop(self.model.parameters(), lr=lr) # noqa
        # self.criterion_loss = torch.nn.MSELoss()  # noqa
        self.criterion_loss = torch.nn.SmoothL1Loss(beta=L1_BETA)  # noqa
        # model.apply(weights)
        self.writer_loss   = SummaryWriter() # noqa
        self.writer_reward = SummaryWriter()

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(train_episodes, 0, train_episodes - 1000 - START_EP).step) # noqa

    def copy_models(self):
        self.target_model.load_state_dict(self.model.state_dict())
        return

    def train(self, replay_memory, curr_episode, batch_size=64):  # model=Q, target_model = Q_tilda
        discount_factor = self.gamma

        if len(replay_memory) < MIN_REPLAY_SIZE:
            return None

        mini_batch         = random.sample(replay_memory, batch_size) # noqa
        current_states     = [transition[0] for transition in mini_batch] # noqa
        actions            = np.array([transition[1] for transition in mini_batch]) # noqa
        rewards            = np.array([transition[2] for transition in mini_batch]) # noqa
        new_current_states = [transition[3] for transition in mini_batch] # noqa



        # cs_data = next(iter(DataLoader(create_2d_iamge(current_states, DEVICE), batch_size=batch_size)))
        # ns_data = next(iter(DataLoader(create_2d_iamge(new_current_states, DEVICE), batch_size=batch_size)))
        # print(cs_data.shape)
        current_qs_list = self.model(create_2d_iamge(current_states, DEVICE))  # noqa
        future_qs_list = self.target_model(create_2d_iamge(new_current_states, DEVICE))  # noqa

        loss_array = np.array([]) # noqa
        actions    = torch.from_numpy(actions).view(-1, 1).to(DEVICE) # noqa

        Y = torch.from_numpy(rewards).to(DEVICE).float() + discount_factor * future_qs_list.max(dim=1)[0] # noqa

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

        # loss_array = np.append(loss_array, loss.item())
        return np.mean(loss.item())

    def decide_action(self, observation, epsilon):
        random_number = np.random.rand()
        # 2. Explore using the Epsilon Greedy Exploration Strategy
        if random_number <= epsilon:
            # Explore
            action = np.random.randint(0, self.actions_shape)
        else:
            # Exploit best known action
            # model dims are (batch, env.observation_space.n)
            predicted = self.model(create_2d_iamge([observation], DEVICE)).flatten()
            action = int(torch.argmax(predicted))
        return action


def read_summary_file(file_path: str = ""):
    with open(file_path, 'r') as f:
        txt = f.read()
        epochs = re.findall(r'After n steps = (\d+)', txt)
        ratios = re.findall(r'final / first reward = (\d+.\d+)', txt)
        loss   = re.findall(r'mean loss = (\d+.\d+)', txt) # noqa
        return np.array(epochs, dtype=int), np.array(ratios, dtype=float), np.array(loss, dtype=float)


def plot_summary_files(files_list):
    if type(files_list) == str and os.path.isdir(files_list):
        files_names = os.listdir(files_list)
        files_list  = [os.path.join(files_list, fname) for fname in files_names] # noqa
    fig, axs = plt.subplots(2)
    for file in files_list:
        epochs, ratios, loss = read_summary_file(file_path=file)
        ratios = np.convolve(ratios, [0.2, 0.2, 0.2, 0.2, 0.2], 'same')
        loss   = np.convolve(loss, [0.2, 0.2, 0.2, 0.2, 0.2], 'same') # noqa
        axs[1].plot(epochs, loss, linewidth=0.3, label=file)
        axs[0].plot(ratios, linewidth=0.3, label=file)
        # axs.plot(ratios, linewidth=0.3, label=file)
        # print(f"{file} - mean last 200 ratio = {np.mean(ratios[-100:])}")
    axs[1].legend(loc="upper right")
    axs[1].set_title('loss')
    axs[0].legend(loc="lower left")
    axs[0].set_title('ratios')
    axs[0].set_ylim([1.3, 1.8])
    # axs.legend(loc="lower left")
    # axs.set_title('ratios')
    # axs.set_ylim([1.3, 1.8])


    axs[0].axhline(1.62)
    # axs.axhline(1.62)
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
    np.random.seed(28)
    # target_ratio = np.random.uniform(1.3, 1.8)

    f = open(OUTPUT_FILE, "w")
    f.write('with mat multiplication, ratio by amount, L2 Loss, RMSprop loss, no TMF-UPRO actions, check if ratio helps\n') # noqa
    epsilon     = 1      # noqa - Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = MAX_EPSILON      # noqa - You can't explore more than 100% of the time
    min_epsilon = 0.001   # noqa - At a minimum, we'll always explore 1% of the time
    decay       = 0.005   # noqa
    lr          = 0.001  # noqa - learning rate
    change_perc = 0.05
    gamma = 0.8
    best_ratio = 1.6

    # Data
    UPRO_values = read_csv_files('../utils_2/SPY.csv') # noqa
    TMF_values = read_csv_files('../utils_2/TYX.csv')   # noqa

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    env = stock_state.env(init_UPRO_val=UPRO_values[0], init_TMF_val=TMF_values[0], num_of_actions=NUM_OF_ACTIONS - 1)
    obs = env.observation()
    env_agent = agent(lr=lr, gamma=gamma, train_episodes=TRAIN_EPISODES)  # noqa - state_space_shape = 3x2 + 1(ration dis) , action_space_shape=3*3*2 + 1

    replay_memory = deque(maxlen=MAX_RB_LENGTH)

    steps_to_update_target_model = 0
    j = 0
    for episode in tqdm.tqdm(range(START_EP, TRAIN_EPISODES)):
        # environment
        env = stock_state.env(init_UPRO_val=UPRO_values[0], init_TMF_val=TMF_values[0], num_of_actions=NUM_OF_ACTIONS-1)
        total_training_rewards = 0
        first_reward           = env.reward() # noqa
        mean_loss = []
        for i, (upro, tmf) in enumerate(zip(UPRO_values[1:], TMF_values[1:])): # noqa
            j += 1
            if (np.mod(j, 5) != 0) and (1-change_perc < upro/UPRO_values[i] < 1+change_perc) and (1-change_perc < tmf/TMF_values[i] < 1+change_perc): # noqa
                continue


            observation = env.observation()
            env.update_stock_values(UPRO_val=upro, TMF_val=tmf)

            # 2. Explore using the Epsilon Greedy Exploration Strategy
            action = env_agent.decide_action(observation=observation.to(DEVICE), epsilon=epsilon)
            new_observation, reward = env.step(action)
            replay_memory.append([observation.to(DEVICE), action, reward, new_observation.to(DEVICE)])

            # 3. Update the Main Network using the Bellman Equation
            if len(replay_memory) >= MIN_REPLAY_SIZE:
                mean_loss.append(env_agent.train(replay_memory=replay_memory, curr_episode=episode))
                steps_to_update_target_model += 1

            total_training_rewards += reward

            if steps_to_update_target_model >= 100:
                # print('Copying main network weights to the target network weights')
                env_agent.copy_models()
                steps_to_update_target_model = 0

        reward = env.reward(final_iteration=True)
        if mean_loss != []: # noqa
            f.write(env.print_env())
            f.write(f'After n steps = {episode}: final / first reward = {float(reward) / float(first_reward) :.2f}\n')
            f.write(f'mean loss = {np.mean(mean_loss) :.2f}    epsilon = {epsilon:.3f}\n\n\n')
            # Save model weights
            torch.save(env_agent.model.state_dict(), os.path.join(os.path.dirname(__file__), "model.pth"))
            if float(reward) / float(first_reward) > best_ratio:
                best_ratio = float(reward) / float(first_reward)
                torch.save(env_agent.model.state_dict(), os.path.join(os.path.dirname(__file__), "best_model.pth"))
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        #     tensorboard
        env_agent.writer_reward.add_scalar("reward ratio", float(reward) / float(first_reward), episode)
        if use_lr_decay:
            env_agent.lr_scheduler.step()

    f.close()
    env_agent.writer_loss.flush()
    env_agent.writer_loss.close()
    env_agent.writer_reward.flush()
    env_agent.writer_reward.close()


if __name__ == '__main__':
    if RUN_MAIN:
        main()
    # file = OUTPUT_FILE
    # plot_summary_files(files_list=[file])

