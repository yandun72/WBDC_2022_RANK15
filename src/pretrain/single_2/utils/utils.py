import os
import random
import torch
import numpy as np
import argparse
import logging
def setup_logging(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='./pretrain.log',
                        filemode='w',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def setup_logging_continue(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='./pretrain.log',
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
        
