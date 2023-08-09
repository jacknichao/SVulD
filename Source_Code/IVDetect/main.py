import dgl
import sys
from dataset import BigVulDatasetIVDetect
from model import IVDetect
from pathlib import Path
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str((Path(__file__).parent.parent.parent)))
from utils import debug, get_run_id, processed_dir, get_metrics_logits, cache_dir, set_seed, result_dir, get_dir
from utils.dclass import BigVulDataset
from utils.my_log import LogWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import pandas as pd


def evaluate(model, val_dl, val_ds, logger, args):
    model.eval()
    with torch.no_grad():
        all_pred = torch.empty((0, 2)).long().to(args.device)
        all_true = torch.empty((0)).long().to(args.device)
        for val_batch in tqdm(val_dl, total=len(val_dl), desc='Validing...'):
            val_batch = val_batch.to(args.device)
            val_labels = dgl.max_nodes(val_batch, "_LABEL").long()
            val_logits = model(val_batch, val_ds)
            all_pred = torch.cat([all_pred, val_logits])
            all_true = torch.cat([all_true, val_labels])
        val_mets = get_metrics_logits(all_true, all_pred)
    return val_mets


def test(model, test_dl, test_ds, logger, args):
    dataset2id = {
        'bigvul': 'balanced',
    }
    balanced_path = result_dir() / f"ivdetect/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    path = balanced_path

    model.load_state_dict(torch.load(path))
    model.eval()
    all_pred = torch.empty((0, 2)).long().to(args.device)
    all_true = torch.empty((0)).long().to(args.device)
    vecs = torch.empty((0)).float().to(args.device)
    with torch.no_grad():
        for test_batch in tqdm(test_dl, total=len(test_dl)):
            test_batch = test_batch.to(args.device)
            test_labels = dgl.max_nodes(test_batch, "_LABEL").long()
            test_logits, vec = model(test_batch, test_ds)
            all_pred = torch.cat([all_pred, test_logits])
            all_true = torch.cat([all_true, test_labels])
            vecs = torch.cat([vecs, vec])
        test_mets = get_metrics_logits(all_true, all_pred)
    if logger:
        logger.test(test_mets)
    else:
        print(test_mets)
    return test_mets


def train(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, logger, args):
    # %% Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for batch in tqdm(train_dl, total=len(train_dl), desc='Training...'):
            # Training
            model.train()
            batch = batch.to(args.device)
            logits = model(batch, train_ds)
            labels = dgl.max_nodes(batch, "_LABEL").long()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_mets = get_metrics_logits(labels, logits)

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


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    configs = json.load(open('./config.json'))
    for item in configs:
        args.__dict__[item] = configs[item]
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"args.device={args.device}")

    dataset = args.dataset
    cache_path = cache_dir() / 'data' / dataset / 'data.pkl'

    df = pd.read_pickle(cache_path)
    train_df = df[df.partition == 'train']
    valid_df = df[df.partition == 'valid']
    test_df = df[df.partition == 'test']

    test_ds = BigVulDatasetIVDetect(df=test_df, partition="test", dataset=dataset)

    dl_args = {"drop_last": False, "shuffle": False, "num_workers": 12}
    test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)

    dev = args.device
    model = IVDetect(input_size=args.input_size, hidden_size=args.hidden_size)
    model.to(dev)

    set_seed(args)
    if args.do_train:
        print("start training .......")
        train_ds = BigVulDatasetIVDetect(df=train_df, partition="train", dataset=dataset)
        val_ds = BigVulDatasetIVDetect(df=valid_df, partition="valid", dataset=dataset)
        dl_args = {"drop_last": False, "shuffle": True, "num_workers": 0}
        train_dl = GraphDataLoader(train_ds, batch_size=args.train_batch_size, **dl_args)
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 0}
        val_dl = GraphDataLoader(val_ds, batch_size=args.test_batch_size, **dl_args)
        args.val_every = int(len(train_dl))
        args.log_every = int(len(train_dl) / 10)

        logger = LogWriter(
            model, args, path=get_dir(result_dir() / f"ivdetect/{args.dataset}/balanced")
        )
        debug(args)
        logger.info(args)
        train(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
              test_dl=test_dl, test_ds=test_ds, logger=logger, args=args)
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)

    if args.do_test:
        print("start testing .......")
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=False)


if __name__ == '__main__':
    main()