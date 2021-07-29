import os, argparse, sys, yaml
from torch.nn.functional import threshold
from torch.backends import cudnn
import torch
import sidekit
from loader import Loader
from solver_encoder import Solver
from scoring import Scoring
import numpy as np
import logging
import logging.config
import pandas as pd

if __name__ == "__main__":
    ### Launch in /home/dzbz0373/Voice/Expe ###
    """VCTK = pd.read_csv("data/VCTK.csv")
    print(VCTK.head())
    print(len(VCTK[VCTK["duration"]>2]))
    Vox1 = pd.read_csv("../Xvectors/db_yaml/voxceleb1_dev.csv")
    print(len(Vox1[Vox1["duration"]>2]))
    Vox2 = pd.read_csv("../Xvectors/db_yaml/voxceleb2_dev.csv")
    print(len(Vox2[Vox2["duration"]>2]))
    exit()"""
    # Fixing seeds 
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Loading params
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/800ep_neck16_meanemb_bs2.yaml', help='yaml conf file for the experiment')
    #parser.add_argument('--logging', type=str, default='logs/300ep_neck8_f.log', help='log file for the experiment')
    args = parser.parse_args()

    #logging param
    """LOG_CONFIG = {'version':1,
              'handlers':{'console':{'class':'logging.StreamHandler'}},
              'root':{'handlers':('console'), 
                      'level':'DEBUG'}}
    logging.config.dictConfig(LOG_CONFIG)"""

    log_format = logging.Formatter('[%(asctime)s] \t %(message)s')
    log = logging.getLogger()                                  
    log.setLevel(logging.INFO)  

    EXPE_NAME = args.config.split("/")[1].strip(".yaml")
    with open(args.config, "r") as ymlfile:
        config = yaml.full_load(ymlfile)
    logging.info("Config loaded from :"+args.config)
    config["EXPE_NAME"]=EXPE_NAME

    # Loading model encoder
    device = torch.device("cuda")
    checkpoint = torch.load(config["model"]["encoder_dir"], map_location=device)
    speaker_number = checkpoint["speaker_number"]
    model_archi = checkpoint["model_archi"]
    Encoder = sidekit.nnet.xvector.Xtractor(speaker_number, model_archi=model_archi, loss=checkpoint["loss"])
    Encoder.load_state_dict(checkpoint["model_state_dict"])
    Encoder = Encoder.eval().cuda().to(device)
    logging.info("Encoder loaded from : "+config["model"]["encoder_dir"] )


    # Generating Dataloader
    """config["data"]["dataset"]="voxceleb1_dev"
    config["data"]["rootdir"]="/ssd/data/VoxCeleb/vox1_dev_wav/wav"
    config["data"]["dataset_yaml"]="/home/dzbz0373/Voice/Expe/data/VoxCeleb1_DS_5s_clean.yaml"
    config["data"]["file_extention"]="wav"
    #config["data"]["train"]["users"]=[0,1089]
    #config["data"]["val"]["users"]=[0,1089]
    config["data"]["test"]["users"]=[1089,1210]"""
    loader =  Loader(config["data"])
    train_loader, val_loader, test_loader = loader.get_dataloader("train"), loader.get_dataloader("val"), loader.get_dataloader("test")
    logging.info("Data loaded from : "+config["data"]["dataset"]+" Size of loaders {}, {}, {}".format(train_loader.__len__(),val_loader.__len__(), test_loader.__len__()))

    #Initiating scoring modules
    test_scorer = Scoring(test_loader, loader.get_dataset("test"), device, name="test", dataset_name=config["data"]["dataset"], n_uttrs=1)
    test_scorer.extract_embeddings(Encoder)
    logging.info("EER on test set : "+str(test_scorer.compute_EER())+" %")
    
    train_scorer = Scoring(train_loader, loader.get_dataset("train"), device, name="train", dataset_name=config["data"]["dataset"])
    train_scorer.extract_embeddings(Encoder)
    logging.info("EER on train set : "+str(train_scorer.compute_EER())+" %")

    val_scorer = Scoring(val_loader, loader.get_dataset("val"), device, name="val", dataset_name=config["data"]["dataset"], n_uttrs=1)
    val_scorer.extract_embeddings(Encoder)
    logging.info("EER on val set : "+str(val_scorer.compute_EER())+" %")
    
    # Initiating Solver
    config["model"]["from_loading"] = True
    solver = Solver((train_loader, val_loader, test_loader), config, Encoder, scorers = (train_scorer, val_scorer, test_scorer), charge_iteration=0)
    logging.info("solver initialized")

    """solver.tar_eval("train", 0.01)
    solver.tar_eval("val", 0.01)
    solver.tar_eval("eval", 0.01)

    solver.tar_eval("train", 0.0182)
    solver.tar_eval("val", 0.0229)
    solver.tar_eval("eval", 0.0432)
    """
    #solver.evaluation("train")
    #solver.evaluation("val")
    #solver.evaluation("eval")

    #solver.tar_eval("train", -1)
    #solver.tar_eval("val", -1)
    #solver.tar_eval("eval", -1)
    """logging.info("TAR for 1%")
    solver.tar_eval("train", 0.01)
    solver.tar_eval("val", 0.01)
    solver.tar_eval("eval", 0.01)
    logging.info("TAR for 0.1%")
    solver.tar_eval("train", 0.001)
    solver.tar_eval("val", 0.001)
    solver.tar_eval("eval", 0.001)"""

    #solver.evaluation("train")
    #solver.evaluation("val")
    solver.evaluation("eval")
    logging.info("### Evaluation Finished ###")


