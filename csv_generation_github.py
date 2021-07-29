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
    parser.add_argument('--config', type=str, default='configs/1ep.yaml', help='yaml conf file for the experiment')
    #parser.add_argument('--logging', type=str, default='logs/300ep_neck8_f.log', help='log file for the experiment')
    args = parser.parse_args()

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

    dataset = "VoxCeleb2"
    # Generating Dataloader
    config["data"]["dataset"]="voxceleb1_dev"
    config["data"]["rootdir"]="/ssd/data/VoxCeleb/vox1_dev_wav/wav"
    config["data"]["dataset_yaml"]="/home/dzbz0373/Voice/Expe/data/VoxCeleb1_DS_5s_clean.yaml"
    config["data"]["file_extention"]="wav"
    #config["data"]["train"]["users"]=[0,1089]
    #config["data"]["val"]["users"]=[0,1089]
    config["data"]["test"]["users"]=[0,1210]
    
    loader =  Loader(config["data"])
    train_loader, val_loader, test_loader = loader.get_dataloader("train"), loader.get_dataloader("val"), loader.get_dataloader("test")
    logging.info("Data loaded from : "+config["data"]["dataset"]+" Size of loaders {}, {}, {}".format(train_loader.__len__(),val_loader.__len__(), test_loader.__len__()))

    test_df = loader.test_df
    test_df.columns.values[0] = "dataset"
    test_df.loc[:,0] = str(dataset)
    test_df = test_df.drop(["speaker_idx",0], axis=1)
    if(dataset=="VoxCeleb2"):test_df["gender"] = test_df["gender"].str[1]
    #print(test_df.head())
    test_df.to_csv("csv_for_github/"+dataset+"_test.csv", index = False)
    
    """train_df = loader.train_df
    train_df.columns.values[0] = "dataset"
    train_df.loc[:,0] = str(dataset)
    train_df['duration'] = 2.04
    train_df = train_df.drop(["speaker_idx",0], axis=1)
    if(dataset=="VoxCeleb2"):train_df["gender"] = train_df["gender"].str[1]
    #print(train_df.head())
    train_df.to_csv("csv_for_github/"+dataset+"_train.csv", index = False)

    val_df = loader.val_df
    val_df.columns.values[0] = "dataset"
    val_df.loc[:,0] = str(dataset)
    val_df['duration'] = 2.04
    val_df = val_df.drop(["speaker_idx",0], axis=1)
    if(dataset=="VoxCeleb2"):val_df["gender"] = val_df["gender"].str[1]
    #print(val_df.head())
    val_df.to_csv("csv_for_github/"+dataset+"_valid.csv", index = False)"""

    logging.info("############################## Generation Finished #####################################")

    for dir,_,files in os.walk("csv_for_github"):
        for file in files:
            df = pd.read_csv("csv_for_github/"+file)
            print(file, "\t", df.shape,"\t", len(set(df["speaker_id"])))
            print(df.head())



