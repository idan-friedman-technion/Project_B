from torch_geometric.nn import GCNConv, Linear
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

DEBUG_MODE = 0
NUM_OF_ACTIONS = 6


def create_graph_batch(list_of_graphs, device):
    print("\n\nsummary:") if DEBUG_MODE else 0
    # print(f"max1 = {max([a.edge_index.shape[1] for a in list_of_graphs])}")
    # print(f"max2 = {max([a.edge_index.max() for a in list_of_graphs])}")
    # print([a.edge_index for a in list_of_graphs])
    batch_size = len(list_of_graphs)
    start_base_nodes = 0
    start_lot_nodes  = (batch_size) * 3  # noqa - 3 nodes for each batch
    x_final_base = torch.tensor([])
    x_final_lot = torch.tensor([])
    edge_index_final = torch.tensor([])
    sum = 0
    for i in range(batch_size):
        graph = list_of_graphs[i]
        x, edge_index = graph.x.clone(), graph.edge_index.clone()
        sum += x.shape[0]
        edge_index_tmp = edge_index.clone()
        edge_index[edge_index_tmp < 3] = edge_index[edge_index_tmp < 3] + start_base_nodes
        edge_index[edge_index_tmp >= 3] = edge_index[edge_index_tmp >= 3] + start_lot_nodes - 3
        if i == 0:
            x_final_base = list_of_graphs[0].x[:3, :]
            x_final_lot = list_of_graphs[0].x[3:, :]
            edge_index_final = edge_index
        else:
            x_final_base = torch.cat((x_final_base, x[:3, :]))
            x_final_lot = torch.cat((x_final_lot, x[3:, :]))
            edge_index_final = torch.cat((edge_index_final, edge_index), dim=1)
        start_base_nodes += 3
        # start_lot_nodes = edge_index_final.max().item() + 1  # set idx to 0 and subtract 3  first nodes
        start_lot_nodes += x.shape[0] - 3  # set idx to 0 and subtract 3  first nodes

    # print(start_lot_nodes) if DEBUG_MODE else 0
    # print(edge_index_final.max()) if DEBUG_MODE else 0
    # print(edge_index_final.shape[1]/2) if DEBUG_MODE else 0
    # print(f"{x_final_base.shape[0]} + {x_final_lot.shape[0]}") if DEBUG_MODE else 0
    # print(sum) if DEBUG_MODE else 0
    return Data(x=torch.cat((x_final_base, x_final_lot)), edge_index=edge_index_final.contiguous())



class LambdaLR(): # noqa
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, batch_size: int=1):
        self.edge_index = [d.edge_index for d in datasets]
        self.x_all = [d.x for d in datasets]
    def __getitem__(self, idx):
        return [self.x_all[idx], self.edge_index[idx]]


class GCN(torch.nn.Module):
    def __init__(self, num_node_features=3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 8)
        self.conv2 = GCNConv(8, num_node_features)
        self.conv3 = GCNConv(num_node_features, 2)
        # self.linear = Linear(19, 19)

    def forward(self, data, batch_size=1):
        x, edge_index = data.x, data.edge_index
        # print(f"x shape begin {x.shape}") if DEBUG_MODE else 0
        # print(f"edge_index begin {edge_index.shape}") if DEBUG_MODE else 0
        x = self.conv1(x, edge_index)
        # print(f"x shape conv1 {x.shape}") if DEBUG_MODE else 0
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        # print(f"x shape final {x.shape}") if DEBUG_MODE else 0
        return x

