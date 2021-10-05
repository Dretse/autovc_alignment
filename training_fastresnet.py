import os, argparse, sys, yaml
from torch.backends import cudnn
import torch
import sidekit
from loader import Loader
import numpy as np
import logging
import logging.config
import configparser
import h5py
import torch
import torch.multiprocessing as mp
from time import time
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['SIDEKIT'] = "libsvm=false"

if __name__ == "__main__":
    ### Launch in /home/dzbz0373/Voice/Expe ###
    LAUNCH_LIUM = False
    # Fixing seeds 
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Loading params
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_fastresnet.yaml', help='yaml conf file for the experiment')
    args = parser.parse_args()


    with open(args.config, "r") as ymlfile:
        config = yaml.full_load(ymlfile)
    logging.info("Config loaded from :"+args.config)
    sidekit.nnet.xtrain(dataset_description=config["dataset_description"],
           model_description=config["model_description"],
           training_description=config["training_description"])
