import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent.parent)))

from utils import processed_dir, dfmp, get_dir, cache_dir, get_node_edges, tokenize
from utils.joern import rdg, get_node_edges, drop_lone_nodes
from utils.dclass import BigVulDataset
import utils.glove as glove
import torch
import dgl
from pathlib import Path
import pickle
import re
import pandas as pd
import networkx as nx
import numpy as np


class BigVulDatasetIVDetect(BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, **kwargs):
        """Init."""
        super(BigVulDatasetIVDetect, self).__init__(**kwargs)
        # Load Glove vectors.
        glove_path = processed_dir() / f"{kwargs['dataset']}/glove_False/vectors.txt"
        self.emb_dict, _ = glove.glove_dict(glove_path)

        # filter large functions
        print(f'{kwargs["partition"]} LOCAL before large:', len(self.df))
        ret = dfmp(
            self.df,
            BigVulDatasetIVDetect._feat_ext_itempath,
            "_id",
            ordr=True,
            desc="Cache features: ",
            workers=32
        )
        self.df = self.df[ret]
        print(f'{kwargs["partition"]} LOCAL after large:', len(self.df))

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        print('stats:', self.stats())
        print(self.df.columns, self.df.shape)
        self.idx2id = pd.Series(self.df._id.values, index=self.df.idx).to_dict()

    def item(self, _id, is_eval=False):
        """Get item data."""
        n, _ = feature_extraction(BigVulDataset.itempath(_id))
        n.subseq = n.subseq.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))
        n.nametypes = n.nametypes.apply(
            lambda x: glove.get_embeddings(x, self.emb_dict, 200)
        )
        n.data = n.data.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))
        n.control = n.control = n.control.apply(lambda x: glove.get_embeddings(x, self.emb_dict, 200))

        asts = []

        def ast_dgl(row, lineid):
            if len(row) == 0:
                return None
            '''
            row example
            [[0, 0, 0, 0, 0, 0], 
             [1, 2, 3, 4, 5, 6], 
             ['int alloc addbyter int output FILE data', 'int output', 'FILE data', '', 'int', 'int output', 'FILE data']]

            '''
            outnode, innode, ndata = row
            g = dgl.graph((outnode, innode))
            g.ndata["_FEAT"] = torch.Tensor(
                np.array(glove.get_embeddings_list(ndata, self.emb_dict, 200))
            )
            g.ndata["_ID"] = torch.Tensor([_id] * g.number_of_nodes())
            g.ndata["_LINE"] = torch.Tensor([lineid] * g.number_of_nodes())
            return g

        for row in n.itertuples():
            asts.append(ast_dgl(row.ast, row.id))

        return {"df": n, "asts": asts}

    def _feat_ext_itempath(_id):
        """Run feature extraction with itempath."""
        n, e = feature_extraction(BigVulDataset.itempath(_id))
        return 0 < len(n) <= 500

    def cache_features(self):
        """Save features to disk as cache."""
        dfmp(
            self.df,
            BigVulDatasetIVDetect._feat_ext_itempath,
            "_id",
            ordr=False,
            desc="Cache features: ",
            workers=32
        )

    def __getitem__(self, idx):
        """Override getitem."""
        _id = self.idx2id[idx]
        n, e = feature_extraction(BigVulDataset.itempath(_id))
        assert len(n), len(e)
        g = dgl.graph(e)
        g.ndata["_LINE"] = torch.Tensor(n["id"].astype(int).to_numpy())
        label = self.get_vul_label(_id)
        g.ndata["_LABEL"] = torch.Tensor([label] * len(n))
        g.ndata["_SAMPLE"] = torch.Tensor([_id] * len(n))
        g.ndata["_PAT"] = torch.Tensor([False] * len(n))

        # Add edges between each node and itself to preserve old node representations
        g = dgl.add_self_loop(g)
        return g


def feature_extraction(filepath):
    """Extract relevant components of IVDetect Code Representation.
    """
    cache_name = "_".join(str(filepath).split("/")[-3:])
    cachefp = get_dir(cache_dir() / f"ivdetect_feat_ext/{BigVulDataset.DATASET}") / Path(cache_name).stem
    try:
        nodes, edges = get_node_edges(filepath)
    except:
        print(filepath)
        return None
    # 1. Generate tokenised subtoken sequences
    subseq = (
        nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
            .groupby("lineNumber")
            .head(1)
    )
    subseq = subseq[~subseq.eq("").any(1)]
    subseq = subseq[subseq.code != " "]
    subseq.lineNumber = subseq.lineNumber.astype(int)
    subseq = subseq.sort_values("lineNumber")
    subseq.code = subseq.code.apply(tokenize)
    subseq = subseq.set_index("lineNumber").to_dict()["code"]

    # 2. Line to AST
    ast_edges = rdg(edges, "ast")
    ast_nodes = drop_lone_nodes(nodes, ast_edges)
    ast_nodes = ast_nodes[ast_nodes.lineNumber != ""]
    ast_nodes.lineNumber = ast_nodes.lineNumber.astype(int)
    ast_nodes["lineidx"] = ast_nodes.groupby("lineNumber").cumcount().values
    ast_edges = ast_edges[ast_edges.line_out == ast_edges.line_in] 
    ast_dict = pd.Series(ast_nodes.lineidx.values, index=ast_nodes.id).to_dict()
    ast_edges.innode = ast_edges.innode.map(ast_dict)
    ast_edges.outnode = ast_edges.outnode.map(ast_dict)
    ast_edges = ast_edges.groupby("line_in").agg({"innode": list, "outnode": list}) 
    ast_nodes.code = ast_nodes.code.fillna("").apply(tokenize)
    nodes_per_line = (
        ast_nodes.groupby("lineNumber").agg({"lineidx": list}).to_dict()["lineidx"]
    )
    ast_nodes = ast_nodes.groupby("lineNumber").agg({"code": list})

    ast = ast_edges.join(ast_nodes, how="inner")
    if ast.empty:
        return [], []
    ast["ast"] = ast.apply(lambda x: [x.outnode, x.innode, x.code], axis=1)
    ast = ast.to_dict()["ast"]
    
    for k, v in ast.items():
        allnodes = nodes_per_line[k]
        outnodes = v[0]
        innodes = v[1]
        lonenodes = [i for i in allnodes if i not in outnodes + innodes]
        parentnodes = [i for i in outnodes if i not in innodes]
        for n in set(lonenodes + parentnodes) - set([0]):
            outnodes.append(0)
            innodes.append(n)
        ast[k] = [outnodes, innodes, v[2]]

    # 3. Variable names and types
    reftype_edges = rdg(edges, "reftype")
    reftype_nodes = drop_lone_nodes(nodes, reftype_edges)
    reftype_nx = nx.Graph()
    reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    reftype_cc = list(nx.connected_components(reftype_nx))
    varnametypes = list()
    for cc in reftype_cc:
        cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
        if sum(cc_nodes["_label"] == "IDENTIFIER") == 0:
            continue
        if sum(cc_nodes["_label"] == "TYPE") == 0:
            continue
        var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].head(1).name.item()
        for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
            varnametypes += [[idrow.lineNumber, var_type, idrow.name]]
    nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])
    nametypes = nametypes.drop_duplicates().sort_values("lineNumber")
    nametypes.type = nametypes.type.apply(tokenize)
    nametypes.name = nametypes.name.apply(tokenize)
    nametypes["nametype"] = nametypes.type + " " + nametypes.name
    nametypes = nametypes.groupby("lineNumber").agg({"nametype": lambda x: " ".join(x)})
    nametypes = nametypes.to_dict()["nametype"]

    # 4/5. Data dependency / Control dependency context
    # Group nodes into statements
    nodesline = nodes[nodes.lineNumber != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
            .groupby("lineNumber")
            .head(1)
    )
    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = rdg(edgesline, "pdg")
    nodesline = drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])

    if len(edgesline) > 0:
        edgesline["etype"] = edgesline.apply(
            lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
        )
        edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
        edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline, edgesline_reverse])
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    uedge = uedge.reset_index()
    if len(uedge) > 0:
        uedge = uedge.pivot("innode", "etype", "outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        uedge.control = uedge.control.apply(
            lambda x: list(x) if isinstance(x, set) else []
        )
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        data = {}
        control = {}

    # Generate PDG
    pdg_nodes = nodesline.copy()
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
    pdg_nodes["ast"] = pdg_nodes.id.map(ast).fillna("")
    pdg_nodes["nametypes"] = pdg_nodes.id.map(nametypes).fillna("")
    pdg_nodes = pdg_nodes[pdg_nodes.id.isin(list(data.keys()) + list(control.keys()))]

    pdg_nodes["data"] = pdg_nodes.id.map(data)
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    pdg_nodes.data = pdg_nodes.data.map(lambda x: ' '.join([subseq[i] for i in x if i in subseq]))
    pdg_nodes.control = pdg_nodes.control.map(lambda x: ' '.join([subseq[i] for i in x if i in subseq]))
    pdg_edges = edgesline.copy()
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    pdg_edges = pdg_edges.dropna()
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    # Cache
    with open(cachefp, "wb") as f:
        pickle.dump([pdg_nodes, pdg_edges], f)
    return pdg_nodes, pdg_edges
