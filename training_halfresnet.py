import os, argparse, sys, yaml
from torch.backends import cudnn
import torch
from loader import Loader
from solver_encoder import Solver
from scoring import Scoring, Multi_scoring
import numpy as np
import logging
import logging.config

import argparse
import configparser
import h5py
import sidekit
import torch
import torch.multiprocessing as mp
from time import time
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#from tensorflow.python.util import deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

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
    parser.add_argument('--config', type=str, default='configs/training_resnet.yaml', help='yaml conf file for the experiment')
    parser.add_argument('--num_expe', type=int, default=0, help='numero of the experiment')
    args = parser.parse_args()


    with open(args.config, "r") as ymlfile:
        config = yaml.full_load(ymlfile)
    logging.info("Config loaded from :"+args.config)
    sidekit.nnet.xtrain(dataset_description=config["dataset_description"].strip(".yaml")+str(args.num_expe)+".yaml",
           model_description=config["model_description"].strip(".yaml")+"l_half.yaml",
           training_description=config["training_description"].strip(".yaml")+str(args.num_expe)+".yaml")


