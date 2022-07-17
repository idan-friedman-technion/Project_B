import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import csv
from utils_3 import stock_state
import tqdm
import re
import datetime
import os
import torch
from torch.utils.data import DataLoader, Dataset

DEBUG_MODE = 0

def create_2d_iamge(list_of_graphs, device):
    batch_size = len(list_of_graphs)
    max_len = np.max([15] + [a.shape[0] for a in list_of_graphs])  # min size for network is 34
    final_graph = torch.zeros(batch_size, 1, 9, max_len).to(device)
    for i, graph in enumerate(list_of_graphs):
        rows_to_add = max_len - graph.shape[0]
        final_graph[i, 0, :, :] = torch.cat((graph, torch.tensor([[0]*9]*rows_to_add).to(device)), 0).t()
    return final_graph



class LambdaLR()
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


class GCN(torch.nn.Module):
    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 2)
        self.conv2 = nn.Conv2d(8, 1, 2)
        self.conv3 = nn.Conv2d(16, 1, 2)
        self.Linear = nn.Conv1d(7, num_classes, 2)
        self.pool = nn.MaxPool2d((1, 2))

    def forward(self, x):
        print(f"original - {x.shape}") if DEBUG_MODE else 0
        x = self.conv1(x)
        print(f"conv1 - {x.shape}") if DEBUG_MODE else 0
        x = self.pool(F.relu(x))
        print(f"pool1 - {x.shape}") if DEBUG_MODE else 0
        x = self.conv2(x)
        print(f"conv2 - {x.shape}") if DEBUG_MODE else 0
        # x = self.conv3(x)
        # print(f"conv3 - {x.shape}") if DEBUG_MODE else 0
        x = self.pool(F.relu(x))
        print(f"pool3 - {x.shape}") if DEBUG_MODE else 0
        # x = torch.flatten(x, 1)
        # print(x.shape)
        # x = self.Linear(x)
        x = self.Linear(torch.squeeze(x, dim=1))
        print(f"final - {x.shape}") if DEBUG_MODE else 0
        return torch.mean(x, dim=2)

