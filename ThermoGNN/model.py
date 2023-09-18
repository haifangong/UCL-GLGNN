import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GraphConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, GraphNorm, global_add_pool, global_max_pool, GlobalAttention

from ThermoGNN.utils.fds import FDS


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


class GNN(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.fc1 = nn.Linear(60, 200)
        self.fc2 = nn.Linear(200, 200)
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            in_dim = input_dim if layer == 0 else emb_dim
            if gnn_type == "gin":
                self.gnns.append(GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                                       nn.Linear(emb_dim, emb_dim))))
            elif gnn_type == "gcn":
                self.gnns.append(GraphConv(in_dim, emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(in_dim, emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(in_dim, emb_dim))
            else:
                raise ValueError("Invalid GNN type.")

    def forward(self, x, edge_index, mut_res_idx, edge_attr=None):
        h_list = [x]
        mut_site = []
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if layer == self.num_layer - 1:
                # remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
            if len(h_list) == 2:
                previous_mut_site_feature = h_list[-2][mut_res_idx]
                current_mut_site_feature = h_list[-1][mut_res_idx]
                # print(previous_mut_site_feature.shape, current_mut_site_feature.shape)
                h_feature = self.fc1(previous_mut_site_feature)
                h_list[-1][mut_res_idx] = h_feature + current_mut_site_feature
            if len(h_list) == 3:
                previous_mut_site_feature = h_list[-2][mut_res_idx].squeeze(0)
                current_mut_site_feature = h_list[-1][mut_res_idx].squeeze(0)

                h_feature = self.fc2(previous_mut_site_feature) + current_mut_site_feature
                h_list[-1][mut_res_idx] = h_feature.unsqueeze(0)
            # mut_site.append()

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)

        return node_representation


# orthogonal initialization
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    # orthogonal initialization of gru weights
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + model.hidden_size], gain=gain)


def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)

    # positive forget gate bias (Jozefowicz es at. 2015)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4: l // 2].data.fill_(1.0)
        hh_b[l // 4: l // 2].data.fill_(1.0)


class GraphGNN(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK="last", drop_ratio=0, graph_pooling="attention",
                 gnn_type="gat", concat_type='lstm', fds=False, feature_level='both', contrast_curri=False) -> object:
        super(GraphGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.concat_type = concat_type
        self.feature_level = feature_level
        self.contrast_curri = contrast_curri
        self.global_local_att0 = nn.Linear(400, 200)
        self.global_local_att1 = nn.Linear(400, 200)

        if self.concat_type == 'lstm':
            # self.lstm_node = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1)
            self.lstm_graph = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1)
            # init_lstm_orth(self.lstm_node)
            # init_lstm_orth(self.lstm_graph)
            # self.fc = nn.Sequential(
            #     nn.Linear(2*self.emb_dim, 2*self.emb_dim // 32, bias=False),
            #     nn.Tanh(),
            #     nn.Linear(2*self.emb_dim // 32, self.out_dim, bias=False),
            # )
            # if self.feature_level == 'both':
            #     self.fc = nn.Linear(2*self.emb_dim, self.out_dim)
            # else:
            self.fc = nn.Linear(self.emb_dim, self.out_dim)

        elif self.concat_type == 'bilstm':
            self.lstm_graph = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1,
                                      bidirectional=True)
            init_lstm_orth(self.lstm_graph)
            self.fc = nn.Linear(2 * self.emb_dim, self.out_dim)

        elif self.concat_type == 'gru':
            self.lstm = nn.GRU(input_size=300, hidden_size=300, num_layers=1)
            init_gru_orth(self.lstm)
            self.fc = nn.Linear(2 * self.emb_dim, self.out_dim)

        else:
            if self.feature_level == 'global-local':
                self.fc = nn.Sequential(
                    nn.Linear(4 * self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                    nn.Linear(self.emb_dim, self.out_dim))
            else:
                self.fc = nn.Sequential(
                    nn.Linear(4 * self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                    nn.Linear(self.emb_dim, self.out_dim))

        if fds:
            self.dir = True
            self.FDS = FDS(4 * self.emb_dim)
        else:
            self.dir = False

        self.gnn = GNN(num_layer, input_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward_once(self, x, edge_index, batch, mut_res_idx):
        mut_res_idx = torch.tensor([mut_res_idx]).cuda()

        node_representation = self.gnn(x, edge_index, mut_res_idx)

        graph_rep = self.pool(node_representation, batch)

        mut_node_rep = node_representation[mut_res_idx].squeeze(0)

        return graph_rep, mut_node_rep

    def forward(self, data, epoch=0):
        wide_res_idx = []
        mut_res_idx = []
        wt_idx = 0
        for i in range(len(data.wide_res_idx)):
            wide_res_idx.append(data.wide_res_idx[i].item() + wt_idx)
            wt_idx += data.wt_count[i].item()

        mut_idx = 0
        for i in range(len(data.mut_res_idx)):
            mut_res_idx.append(data.mut_res_idx[i].item() + mut_idx)
            mut_idx += data.mut_count[i].item()

        graph_rep_be, node_rep_be = self.forward_once(data.x_s, data.edge_index_s, data.x_s_batch, wide_res_idx)
        graph_rep_af, node_rep_af = self.forward_once(data.x_t, data.edge_index_t, data.x_t_batch, mut_res_idx)

        if self.concat_type == 'concat':
            if self.feature_level == 'global-local':
                x = torch.cat([graph_rep_be, node_rep_be, graph_rep_af, node_rep_af], dim=1)
            elif self.feature_level == 'global-local-att':
                # print(graph_rep_be.shape)
                # print(node_rep_be.shape)
                before_rep = self.global_local_att0(torch.cat([graph_rep_be, node_rep_be], dim=1))
                fuse1 = before_rep.mul(node_rep_be)
                before_f = graph_rep_be + fuse1
                after_rep = self.global_local_att1(torch.cat([graph_rep_af, node_rep_af], dim=1))
                fuse2 = after_rep.mul(node_rep_af)
                after_f = graph_rep_af + fuse2
                x = torch.cat([before_f, after_f], dim=1)
            elif self.feature_level == 'global':
                x = torch.cat([graph_rep_be, graph_rep_af], dim=1)
            elif self.feature_level == 'local':
                x = torch.cat([node_rep_be, node_rep_be], dim=1)

            if self.dir:
                smooth_x = x
                x = self.FDS.smooth(smooth_x, data.y, epoch)
        else:
            graph_rep_0, graph_rep_1 = graph_rep_be.unsqueeze_(0), graph_rep_af.unsqueeze_(0)
            lstm_graph_in = torch.cat((graph_rep_0, graph_rep_1), dim=0)
            # lstm_node_in = torch.cat((node_rep_1, node_rep_0), dim=0)

            # node_t1, _ = self.lstm_node(lstm_node_in)
            # node = node_t1[-1]
            graph_t1, (_, _) = self.lstm_graph(lstm_graph_in)
            x = graph_t1[-1]

            if self.dir:
                smooth_x = x
                x = self.FDS.smooth(smooth_x, data.y, epoch)

        if self.dir:
            x = self.fc(x)
            return torch.squeeze(x), smooth_x
        elif self.contrast_curri:
            similarity_list = []
            for i in range(node_rep_be.shape[0]):
                similarity_list.append(cosine_similarity(np.asarray(node_rep_be[i].cpu().detach()),
                                                         np.asarray(node_rep_af[i].cpu().detach())))
            x = self.fc(x)
            return torch.squeeze(x), similarity_list
        else:
            x = self.fc(x)
            return torch.squeeze(x)
