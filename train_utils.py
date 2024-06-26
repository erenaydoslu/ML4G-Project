import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import AGNNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from matplotlib import pyplot as plt
import numpy as np


import numpy as np

def create_forecasting_dataset(graph_signals, splits: list, pred_horizen: int, obs_window: int, verbose: int = 0):
    data = graph_signals
    num_samples = graph_signals.shape[0]
    
    if splits is not None:
        max_idx_trn = int(num_samples * splits[0])
        max_idx_val = int(num_samples * sum(splits[:-1]))
        split_idx = np.split(np.arange(num_samples), [max_idx_trn, max_idx_val])
        data_dict = {}
        data_type = ['trn', 'val', 'tst']
    else:
        split_idx = [np.arange(num_samples)]
        data_dict = {}
        data_type = ['trn']

    for i in range(len(data_type)):
        split_data = data[split_idx[i], :]
        data_points = []
        targets = []

        for j in range(len(split_idx[i]) - obs_window - pred_horizen + 1):
            data_points.append(split_data[j:j+obs_window, :].transpose(1, 0))
            targets.append(split_data[j+obs_window:j+obs_window+pred_horizen, :].transpose(1, 0))

        data_dict[data_type[i]] = {
            'data': np.stack(data_points, axis=0),
            'labels': np.stack(targets, axis=0)
        }
    if verbose == 1:
        print("Dataset has been created.")
        print("-------------------------")
        print(f"{data_dict['trn']['data'].shape[0]} train data points")
        if splits is not None:
            print(f"{data_dict['val']['data'].shape[0]} validation data points")
            print(f"{data_dict['tst']['data'].shape[0]} test data points")

    return data_dict




def concatentate_static_features(data, static_data):
    data_list = []
    for i in range(data.shape[0]):
        data_list.append(np.concatenate([data[i],static_data], axis = 1))
    return np.array(data_list)