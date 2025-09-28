import os
import setproctitle
import argparse
from trainer import Trainer
from utils import *
import torch
import numpy as np

print('pid:', os.getpid())

def main():
    ### set process name
    setproctitle.setproctitle('BNbench')

    ### set hyperparameters
    parser = argparse.ArgumentParser(description='Task Aware Relation Graph for Few-shot Chemical Property Prediction')
    # general hyperparameters
    parser.add_argument('--model', type=str, default='MRCGNN', choices=['MSTE', 'MLP', 'Decagon', 'TIGER', 'SSI-DDI', 'MRCGNN', 'SAGAN'])
    parser.add_argument('--name', default='testrun', help='Set run name for saving/restoring models')

    ### dataset setting
    parser.add_argument('--dataset', type=str, default='drugbank', choices=['drugbank', 'twosides'])
    parser.add_argument('--dataset_type', type=str, default='cluster', choices=['random', 'cluster']) ### exchange random and sail

    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument('--lbl_smooth',	type=float,     default=0.0,	help='Label Smoothing') ### usually 0-1
    parser.add_argument("--epoch", type=int, default=1, help="training epoch")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--use_feat', default=1, type=bool, help='Whether to use drug feature')

    parser.add_argument('--seed', default=124, type=int, help='Seed for randomization')
    parser.add_argument('--eval_skip', default=1, type=int, help='Evaluate every x epochs')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')
    
    # KGE models
    parser.add_argument('--kge_dim', type=int, default=200, help='hidden dimension.')
    parser.add_argument('--kge_gamma', type=int, default=1, help='gamma parameter.')
    parser.add_argument('--kge_dropout', type=float, default=0, help='dropout rate.') ### DDI best 0
    parser.add_argument('--kge_loss', type=str, default='BCE_mean',  help='loss function')

    # MLP model
    parser.add_argument('--mlp_dropout', type=float, default=0.1, help='dropout rate.')
    parser.add_argument('--mlp_dim', type=int, default=200, help='hidden dimension.')

    ### Decagon model decagon_drop
    parser.add_argument('--decagon_dim', type=int, default=200, help='hidden dimension.')
    parser.add_argument('--decagon_drop', type=float,	default=0.1, help='Dropout to use in Decagon model')

    ### set basic configurations
    args = parser.parse_args()

    ### set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model in ['MSTE']:
        args.use_feat = 0
        if args.dataset in ['twosides']:
            args.batch_size = 128
    
    if args.model == 'SAGAN':
        args.adversarial = 1
    else:
        args.adversarial = 0

    args.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"

    ### Training step in the trainer
    trainer = Trainer(args)
    trainer.run()

if __name__ == "__main__":
    main()
