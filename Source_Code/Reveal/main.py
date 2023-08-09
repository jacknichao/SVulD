import os
import sys
from torch.optim import Adam
from graph_dataset import create_dataset, DataSet
from pathlib import Path
sys.path.append(str((Path(__file__).parent)))
sys.path.append(str((Path(__file__).parent.parent.parent)))
import os
cur_dir = os.getcwd()
pkg_rootdir = os.path.dirname(os.path.dirname(cur_dir))
print(pkg_rootdir)
if pkg_rootdir not in sys.path:
    sys.path.append(pkg_rootdir)

from models.reveal.model import MetricLearningModel
from trainer import train, show_representation
from utils import debug, get_run_id, processed_dir,cache_dir, set_seed, result_dir, get_dir
import numpy as np
import random
import torch
import warnings

warnings.filterwarnings('ignore')
import argparse
from tsne import plot_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()
    seed = 123456
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    
    ggnn_bin_path = cache_dir()/f"ggnn_output/{args.dataset}/balanced/ggnn_output.bin"
    ggnn_output = torch.load(ggnn_bin_path)

 
    dataset = create_dataset(
        ggnn_output=ggnn_output,
        batch_size=128,
        output_buffer=sys.stderr
    )
    num_epochs = 200
    dataset.initialize_dataset(balance=True) 

    print(dataset.hdim, end='\t')
    model = MetricLearningModel(input_dim=dataset.hdim, hidden_dim=256)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
  
    show_representation(model, dataset.get_next_test_batch, dataset.initialize_test_batches(), 0,
                        args.dataset + '-after-training-triplet')

    pass
