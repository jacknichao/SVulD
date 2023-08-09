import dgl
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
# error:no module named 'models'
import os
cur_dir = os.getcwd()
pkg_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
print(pkg_rootdir)
if pkg_rootdir not in sys.path:
    sys.path.append(pkg_rootdir)
from models.devign.dataset import BigVulDatasetDevign
from models.reveal.ggnn.model import GGNNSum
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')
from utils import debug, get_run_id, processed_dir, get_metrics_probs_bce,get_metrics_logits,cache_dir, set_seed, result_dir, get_dir
from utils.dclass import BigVulDataset
from utils.my_log import LogWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2,3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import pandas as pd


def evaluate(model, val_dl, val_ds, logger, args):
    model.eval()
    with torch.no_grad():
        all_probs = torch.empty((0)).float().to(args.device)
        all_logits = torch.empty((0)).float().to(args.device)
        all_true = torch.empty((0)).float().to(args.device)
        for val_batch in tqdm(val_dl, total=len(val_dl), desc='Validing...'):
            val_batch = val_batch.to(args.device)
            val_labels = dgl.max_nodes(val_batch, "_LABEL")
            val_probs, val_logits = model(val_batch, val_ds)
            all_true = torch.cat([all_true, val_labels])
            all_probs = torch.cat([all_probs, val_probs])
            all_logits = torch.cat([all_logits, val_logits])

        val_mets = get_metrics_probs_bce(all_true, all_probs, all_logits)
    return val_mets


def test(model, test_dl, test_ds, logger, args):
    logger.load_best_model()
    model.eval()
    all_pred = torch.empty((0)).float().to(args.device)
    all_probs = torch.empty((0)).float().to(args.device)
    all_logits = torch.empty((0)).float().to(args.device)
    all_true = torch.empty((0)).float().to(args.device)
    with torch.no_grad():
        for test_batch in test_dl:
            test_batch = test_batch.to(args.device)
            test_labels = dgl.max_nodes(test_batch, "_LABEL")
            test_probs, test_logits = model(test_batch, test_ds)

            all_probs = torch.cat([all_probs, test_probs])
            all_logits = torch.cat([all_logits, test_logits])
            all_true = torch.cat([all_true, test_labels])
        test_mets = get_metrics_probs_bce(all_true, all_probs, all_logits)
    logger.test(test_mets)
    return test_mets


def train(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, logger, args):
    # %% Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    loss_function = nn.BCELoss()
    for epoch in range(args.epochs):
        for batch in tqdm(train_dl, total=len(train_dl), desc='Training...'):
            # Training
            model.train()
            batch = batch.to(args.device)
            probs, logits = model(batch, train_ds)
            labels = dgl.max_nodes(batch, "_LABEL")
            loss = loss_function(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # train_mets = get_metrics_probs_bce(labels, logits)
            train_mets = get_metrics_probs_bce(labels, probs, logits)

            # Evaluation
            val_mets = None
            if logger.log_val():
                val_mets = evaluate(model, val_dl=val_dl, val_ds=val_ds, logger=logger, args=args)
            logger.log(train_mets, val_mets)
            logger.save_logger()

        # Early Stopping
        if logger.stop():
            break
        logger.epoch()

    # Print test results
    test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)


def save_ggnn_output(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, args):
    
    print("***** Running save_ggnn_output *****")

    model.eval()
    
    ID = {
        'bigvul': 'balanced'
    }

    path = result_dir() / f"reveal/{args.dataset}"/ f"{ID[args.dataset]}/best_f1.model"

    model.load_state_dict(torch.load(path))

    with torch.no_grad():
        cache_all = []
        for data_dl, data_ds in [(train_dl, train_ds), (val_dl, val_ds), (test_dl, test_ds)]:
            all_pred = torch.empty((0)).float().to(args.device)
            all_true = torch.empty((0)).float().to(args.device)
            print(len(data_dl), len(data_ds))
            for test_batch in data_dl:
                test_batch = test_batch.to(args.device)
                test_labels = dgl.max_nodes(test_batch, "_LABEL")
                test_logits, ggnn_output = model.save_ggnn_output(test_batch, data_ds)
                all_pred = torch.cat([all_pred, ggnn_output])
                all_true = torch.cat([all_true, test_labels])
            cache_all.append((all_pred, all_true))
        cache_path = get_dir(cache_dir() / f"ggnn_output/{args.dataset}/balanced/")
        torch.save(cache_all, cache_path / 'ggnn_output.bin')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_after_ggnn', default='True', action='store_true')
    args = parser.parse_args()
    configs = json.load(open('./config.json'))
    for item in configs:
        args.__dict__[item] = configs[item]
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    cache_path = cache_dir() / "data/bigvul/data.pkl"

    df = pd.read_pickle(cache_path)
    train_df = df[df.partition == 'train']
    valid_df = df[df.partition == 'valid']
    test_df = df[df.partition == 'test']

    train_ds = BigVulDatasetDevign(df=train_df, partition="train", dataset=dataset)
    val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)

    test_ds = BigVulDatasetDevign(df=test_df, partition="test", dataset=dataset)
    dl_args = {"drop_last": False, "shuffle": True, "num_workers": 6}
    train_dl = GraphDataLoader(train_ds, batch_size=args.train_batch_size, **dl_args)
    dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
    val_dl = GraphDataLoader(val_ds, batch_size=args.test_batch_size, **dl_args)
    test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)
    args.val_every = int(len(train_dl))
    args.log_every = int(len(train_dl) / 5)
    dev = args.device
    model = GGNNSum(input_dim=args.input_size, output_dim=args.hidden_size, num_steps=args.num_steps)
    model.to(dev)

    set_seed(args)
    if args.save_after_ggnn: 
        save_ggnn_output(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
              test_dl=test_dl, test_ds=test_ds, args=args)
        return
    
    logger = LogWriter(
        model, args, path=get_dir(result_dir() / f"reveal/{args.dataset}/balanced")
    )
    debug(args)
    logger.info(args)
    
    if args.do_train:
        train(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
              test_dl=test_dl, test_ds=test_ds, logger=logger, args=args)
    if args.do_test:
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)


if __name__ == '__main__':
    main()