import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import random
from loguru import logger
import date
from model_loader import load_model
import warnings
warnings.filterwarnings("ignore")

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



num_features = {
    'resnet50':2048,
}


def run():
    # Load configuration
    seed_torch()
    seed= 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    args = load_config()
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")
    logger.info(args)

    if args.tag == 'officehome':
        from officehome import load_data
    elif args.tag == 'office':
        from office31 import load_data
    else:
        raise ValueError('Invalid dataset name')
    


    # Load dataset
    query_dataloader, train_s_dataloader, train_t_dataloader, retrieval_dataloader \
     = load_data(args.source, args.target,args.batch_size,args.num_workers)
    
    if args.train:
        date.train(
            train_s_dataloader,
            train_t_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.max_iter,
            args.arch,
            args.lr,
            args.device,
            args.verbose,
            args.topk,
            args.num_class,
            args.evaluate_interval,
            args.tag,
            args.training_source,
            args.training_target,
            num_features[args.arch],
            args.max_iter_target,
            args.gpu,
            args.save_source_statistics,
            args.alpha,
            args.eta,
        )

    else:
        raise ValueError('Error configuration, please check your config, using "train", "resume" or "evaluate".')


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='date_PyTorch')

    # Dataset Office-31
    # webcam : /office/webcam_list.txt
    # amazon : /office/amazon_list.txt
    # dslr : /office/dslr_list.txt
    #parser.add_argument('--tag', type=str, default='office', help="Tag")
    #parser.add_argument('--source', type=str, default='/data/user/da_datasets/office/amazon_list.txt', help="The source dataset")
    #parser.add_argument('--target', type=str, default='/data/user/da_datasets/office/dslr_list.txt', help="The target dataset")
    # the number of classes
    parser.add_argument('--num_class', default=65, type=int,
                        help='Number of classes(default:Office-Home:65, Office-31:31)')

    # Dataset Office-Home
    # Art : /office-home/Art.txt
    # Clipart :/office-home/Clipart.txt
    # Product : /office-home/Product.txt
    # Real_World : /office-home/Real_World.txt
    parser.add_argument('--tag', type=str, default='officehome', help="Tag")
    parser.add_argument('--source', type=str, default='/data/user/da_datasets/office-home/Art.txt', help="The source dataset")
    parser.add_argument('--target', type=str, default='/data/user/da_datasets/office-home/Real_World.txt', help="The target dataset")

    

    parser.add_argument('-c', '--code-length', default=64, type=int,
                        help='Binary hash code length.(default: 64)')
    parser.add_argument('-k', '--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: -1)')
    parser.add_argument('-T', '--max-iter', default=50, type=int,
                        help='Number of iterations for source training.(default: 50)')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-3)')
    parser.add_argument('-w', '--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('-a', '--arch', default='resnet50', type=str,
                        help='CNN architecture.(default: resnet50)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print log.')
    parser.add_argument('--train', action='store_true',
                        help='Training mode.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluation mode.')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='gpu number')
    parser.add_argument('-e', '--evaluate-interval', default=10, type=int,
                        help='Interval of evaluation.(default: 10)')
    parser.add_argument('--training_source', type=bool, default=True, help='training on the source data')
    parser.add_argument('--training_target', type=bool, default=True, help='training on the target data')
    parser.add_argument('--save_source_statistics', type=bool, default=True, help='pre-compute and store source statistics')
    parser.add_argument('--max_iter_target', default=50, type=int, help='Number of iterations on target data.(default: 50)')
    parser.add_argument('--alpha', default=0.035, type=float,
                        help='percentile of cosine similarity scores.(default:0.035)')
    parser.add_argument('--eta', default=0.1, type=float,
                        help='balance parameter .(default:0.1)')
    

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(args.gpu)

    return args


if __name__ == '__main__':
    run()
