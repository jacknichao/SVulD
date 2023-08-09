import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
import torch.nn.functional as f
import dgl


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types=3, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, dataset, cuda=False):
        features = g.ndata['_WORD2VEC']
        edge_types = g.edata["_ETYPE"]
        outputs = self.ggnn(g, features, edge_types)
        g.ndata['GGNNOUTPUT'] = outputs

        h_i = self.unbatch_features(g)
        h_i = torch.stack(h_i)
        h_i_sum = h_i.sum(dim=1)
        ggnn_sum = self.classifier(h_i_sum)
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result,ggnn_sum 

    def save_ggnn_output(self, g, dataset, cuda=False):
        features = g.ndata['_WORD2VEC']
        edge_types = g.edata["_ETYPE"]
        outputs = self.ggnn(g, features, edge_types)
        g.ndata['GGNNOUTPUT'] = outputs

        h_i = self.unbatch_features(g)
        h_i = torch.stack(h_i)
        h_i_sum = h_i.sum(dim=1)
        ggnn_sum = self.classifier(h_i_sum)
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result, h_i_sum

    def unbatch_features(self, g):
        h_i = []
        max_len = -1
        for g_i in dgl.unbatch(g):
            h_i.append(g_i.ndata['GGNNOUTPUT'])
            max_len = max(g_i.number_of_nodes(), max_len)
        for i, k in enumerate(h_i):
            h_i[i] = torch.cat(
                (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                device=k.device)), dim=0)
        return h_i
