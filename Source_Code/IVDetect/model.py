import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from dgl.nn.pytorch import GraphConv, AvgPooling, MaxPooling
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent)))
from treeLstm import TreeLSTM
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class DynamicRNN(nn.Module):
    """Wrapper function to enable packed sequence RNNs.

    Copied from: https://gist.github.com/davidnvq/594c539b76fc52bef49ec2332e6bcd15
    """

    def __init__(self, rnn_module):
        """Init wrapper."""
        super().__init__()
        self.rnn_module = rnn_module

    def forward(self, x, len_x, initial_state=None):
        """
        Forward pass.

        Arguments
        ---------
        x : torch.FloatTensor
                padded input sequence tensor for RNN model
                Shape [batch_size, max_seq_len, embed_size]
        len_x : torch.LongTensor
                Length of sequences (b, )
        initial_state : torch.FloatTensor
                Initial (hidden, cell) states of RNN model.
        Returns
        -------
        A tuple of (padded_output, h_n) or (padded_output, (h_n, c_n))
                padded_output: torch.FloatTensor
                        The output of all hidden for each elements. The hidden of padding elements will be assigned to
                        a zero vector.
                        Shape [batch_size, max_seq_len, hidden_size]
                h_n: torch.FloatTensor
                        The hidden state of the last step for each packed sequence (not including padding elements)
                        Shape [batch_size, hidden_size]
                c_n: torch.FloatTensor
                        If rnn_model is RNN, c_n = None
                        The cell state of the last step for each packed sequence (not including padding elements)
                        Shape [batch_size, hidden_size]
        """
        # First sort the sequences in batch in the descending order of length

        sorted_len, idx = len_x.sort(dim=0, descending=True)
        sorted_x = x[idx]

        # Convert to packed sequence batch
        packed_x = pack_padded_sequence(sorted_x, lengths=sorted_len, batch_first=True)

        # Check init_state
        if initial_state is not None:
            if isinstance(initial_state, tuple):  # (h_0, c_0) in LSTM
                hx = [state[:, idx] for state in initial_state]
            else:
                hx = initial_state[:, idx]  # h_0 in RNN
        else:
            hx = None

        # Do forward pass
        self.rnn_module.flatten_parameters()
        packed_output, last_s = self.rnn_module(packed_x, hx)

        # pad the packed_output
        max_seq_len = x.size(1)
        padded_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_seq_len
        )

        # Reverse to the original order
        _, reverse_idx = idx.sort(dim=0, descending=False)
        padded_output = padded_output[reverse_idx]

        return padded_output, last_s


class GruWrapper(nn.Module):
    """Get last state from GRU."""

    def __init__(
            self, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False
    ):
        """Initilisation."""
        super(GruWrapper, self).__init__()
        self.gru = DynamicRNN(
            nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
            )
        )

    def forward(self, x, x_lens, return_sequence=False):
        """Forward pass."""
        # Load data from disk on CPU
        out, hidden = self.gru(x, x_lens)
        if return_sequence:
            return out, hidden
        out = out[range(out.shape[0]), x_lens - 1, :]
        return out, hidden


class IVDetect(nn.Module):
    """IVDetect model."""

    def __init__(self, input_size, hidden_size, num_conv_layers=3, dropout=0.5):
        """Initilisation."""
        super(IVDetect, self).__init__()
        self.layer_num = num_conv_layers
        self.gru = GruWrapper(input_size, hidden_size)
        self.gru2 = GruWrapper(input_size, hidden_size)
        self.gru3 = GruWrapper(input_size, hidden_size)
        self.gru4 = GruWrapper(input_size, hidden_size)
        self.bigru = nn.GRU(
            hidden_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.treelstm = TreeLSTM(input_size, hidden_size, dropout=0)
        self.gcn = GraphConv(hidden_size, 2)
        self.connect = nn.Linear(hidden_size * 5 * 2, hidden_size)
        self.dropout = dropout
        self.h_size = hidden_size
        self.relu = nn.ReLU()

        self.avg_pool = AvgPooling()
        self.max_pool = MaxPooling()

    def forward(self, g, dataset, e_weights=[]):
        """Forward pass.

        DEBUG:
        import sastvd.helpers.graphs as svdgr
        from importlib import reload
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = BigVulDatasetIVDetect(partition="train", sample=10)
        g = dgl.batch([dataset[0], dataset[1]]).to(dev)

        input_size = 200
        hidden_size = 200
        num_layers = 2

        reload(ivdts)
        model = IVDetect(200, 64).to(dev)
        ret = model(g, dataset)

        """
        # Load data from disk on CPU
        nodes = list(
            zip(
                g.ndata["_SAMPLE"].detach().cpu().int().numpy(),
                g.ndata["_LINE"].detach().cpu().int().numpy(),
            )
        )

        data = dict()
        asts = []
        for sampleid, is_eval in set([(n[0], n[1]) for n in list(
                zip(
                    g.ndata["_SAMPLE"].detach().cpu().int().numpy(),
                    g.ndata["_PAT"].detach().cpu().int().numpy(),
                )
        )]):
            # datasetitem = dataset.item(sampleid, is_eval)
            datasetitem = dataset.item(sampleid, is_eval)
            for row in datasetitem["df"].to_dict(orient="records"):
                data[(sampleid, row["id"])] = row
            asts += datasetitem["asts"]
        asts = [i for i in asts if i is not None]
        asts = dgl.batch(asts).to(self.dev)

        feat = defaultdict(list)
        for n in nodes:
            f1 = torch.Tensor(np.array(data[n]["subseq"]))
            f1 = f1 if f1.shape[0] > 0 else torch.zeros(1, 200)
            f1_lens = len(f1)
            feat["f1"].append(f1)
            feat["f1_lens"].append(f1_lens)

            f3 = torch.Tensor(np.array(data[n]["nametypes"]))
            f3 = f3 if f3.shape[0] > 0 else torch.zeros(1, 200)
            f3_lens = len(f3)
            feat["f3"].append(f3)
            feat["f3_lens"].append(f3_lens)

            f4 = torch.Tensor(np.array(data[n]["data"]))
            f4 = f4 if f4.shape[0] > 0 else torch.zeros(1, 200)
            f4_lens = len(f4)
            feat["f4"].append(f4)
            feat["f4_lens"].append(f4_lens)

            f5 = torch.Tensor(np.array(data[n]["control"]))
            f5 = f5 if f5.shape[0] > 0 else torch.zeros(1, 200)
            f5_lens = len(f5)
            feat["f5"].append(f5)
            feat["f5_lens"].append(f5_lens)

        # Pass through GRU / TreeLSTM
        F1, _ = self.gru(
            pad_sequence(feat["f1"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f1_lens"]).long(),
        )
        F2 = self.treelstm(asts)
        F3, _ = self.gru2(
            pad_sequence(feat["f3"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f3_lens"]).long(),
        )
        F4, _ = self.gru3(
            pad_sequence(feat["f1"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f1_lens"]).long(),
        )
        F5, _ = self.gru4(
            pad_sequence(feat["f1"], batch_first=True).to(self.dev),
            torch.Tensor(feat["f1_lens"]).long(),
        )

        # Fill null values (e.g. line has no AST representation / datacontrol deps)
        F2 = torch.stack(
            [F2[i] if i in F2 else torch.zeros(self.h_size).to(self.dev) for i in nodes]
        )

        F1 = F1.unsqueeze(1)
        F2 = F2.unsqueeze(1)
        F3 = F3.unsqueeze(1)
        F4 = F4.unsqueeze(1)
        F5 = F5.unsqueeze(1)

        feat_vec, _ = self.bigru(torch.cat((F1, F2, F3, F4, F5), dim=1))
        feat_vec = F.dropout(feat_vec, self.dropout)
        feat_vec = torch.flatten(feat_vec, 1)
        feat_vec = self.connect(feat_vec)
        g.ndata["h"] = self.gcn(g, feat_vec)
        batch_pooled = self.avg_pool(g, g.ndata["h"])
        return batch_pooled, self.avg_pool(g, feat_vec)
