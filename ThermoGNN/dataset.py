from collections import Counter

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from ThermoGNN.utils.weights import assign_weights

import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang


def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window


def get_bin_idx(x):
    return max(min(int(x * np.float32(5)), 12), -12)


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        if key == 'wide_nodes':
            return self.x_s.num_nodes
        if key == 'mut_nodes':
            return self.x_t.num_nodes
        else:
            return super().__inc__(key, value, *args)


def load_dataset(graph_dir, split="train", labeled=True, dir=False):

    data_list = []
    num_nodes = 0
    num_edges = 0
    
    cos_file = open('cos.txt', 'w')

    for i, name in enumerate(open(f"data/{split}_names.txt")):
        name = name.strip()
        G_wt = nx.read_gpickle(f"{graph_dir}/{split}/{name}_wt.pkl")
        data_wt = from_networkx(G_wt)
        G_mut = nx.read_gpickle(f"{graph_dir}/{split}/{name}_mut.pkl")
        data_mut = from_networkx(G_mut)

        # cosine_similarity_score = cosine_similarity(data_wt.x[G_wt.graph['mut_pos']], data_mut.x[G_mut.graph['mut_pos']])
        # cos_file.write(str(cosine_similarity_score.item())+' '+str(G_wt.graph['y'])+'\n')
        # print()
        # return

        wt_node_count = data_wt.num_nodes
        mut_node_count = data_mut.num_nodes

        data_direct = PairData(data_wt.edge_index, data_wt.x,
                               data_mut.edge_index, data_mut.x)
        data_direct.wide_res_idx = G_wt.graph['mut_pos']
        data_direct.mut_res_idx = G_mut.graph['mut_pos']
        data_direct.wt_count = wt_node_count
        data_direct.mut_count = mut_node_count

        data_reverse = PairData(data_mut.edge_index, data_mut.x,
                                data_wt.edge_index, data_wt.x)
        data_reverse.wide_res_idx = G_mut.graph['mut_pos']
        data_reverse.mut_res_idx = G_wt.graph['mut_pos']
        data_reverse.wt_count = mut_node_count
        data_reverse.mut_count = wt_node_count

        if labeled:
            data_direct.y = G_wt.graph['y']
            data_reverse.y = -G_mut.graph['y']

        if dir:
            weights = assign_weights("data/datasets/train_data_noisy.txt")
            data_direct.wy = torch.tensor(weights[i])
            data_reverse.wy = torch.tensor(weights[i])

        data_list.append(data_direct)
        data_list.append(data_reverse)
        num_nodes += data_wt.num_nodes
        num_nodes += data_mut.num_nodes
        num_edges += data_wt.num_edges
        num_edges += data_mut.num_edges

    print(f'{split.upper()} DATASET:')
    print(f'Number of nodes: {num_nodes / len(data_list):.2f}')
    print(f'Number of edges: {num_edges / len(data_list):.2f}')
    print(f'Average node degree: {num_edges / num_nodes:.2f}')

    return data_list
