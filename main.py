import os, argparse, sys, yaml
from torch.backends import cudnn
import torch
import sidekit
from loader import Loader
from solver_encoder import Solver
from scoring import Scoring, Multi_scoring
import numpy as np
import logging
import logging.config

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
    parser.add_argument('--config', type=str, default='configs/1ep.yaml', help='yaml conf file for the experiment')
    parser.add_argument('--logging', type=str, default='logs/1ep.log', help='log file for the experiment')
    #parser.add_argument('--config', type=str, default='configs/500ep_neck16_emb10.yaml', help='yaml conf file for the experiment')
    #parser.add_argument('--logging', type=str, default='logs/500ep_neck16_emb10.log', help='log file for the experiment')
    args = parser.parse_args()

    #logging param
    LOG_CONFIG = {'version':1,
              'handlers':{'console':{'class':'logging.StreamHandler'},
                          'file':{'class':'logging.FileHandler',
                                  'filename':args.logging}},
              'root':{'handlers':('console', 'file'), 
                      'level':'DEBUG'}}
    logging.config.dictConfig(LOG_CONFIG)

    EXPE_NAME = args.config.split("/")[1].strip(".yaml")
    with open(args.config, "r") as ymlfile:
        config = yaml.full_load(ymlfile)
    logging.info("Config loaded from :"+args.config)
    config["EXPE_NAME"]=EXPE_NAME
    with open("data/"+config["dataset"]+".yaml", "r") as ymlfile:
        dataset = yaml.full_load(ymlfile)
    logging.info("Dataset loaded :"+config["dataset"])

    dataset["data_path"]=dataset["rootdir_lium"] if LAUNCH_LIUM else dataset["rootdir_orange"]
    if(LAUNCH_LIUM):logging.info("Experiment launched on lium cluster")
    else:logging.info("Experiment launched on Orange PC")
    
    # Loading model encoder
    device = torch.device("cuda")
    checkpoint = torch.load(config["model"]["encoder_dir"], map_location=device)
    speaker_number = checkpoint["speaker_number"]
    model_archi = checkpoint["model_archi"]
    model_archi = model_archi["model_type"]
    Encoder = sidekit.nnet.xvector.Xtractor(speaker_number, model_archi=model_archi, loss=checkpoint["loss"])
    Encoder.load_state_dict(checkpoint["model_state_dict"])
    Encoder = Encoder.eval().cuda().to(device)
    logging.info("Encoder loaded from : "+config["model"]["encoder_dir"] )


    # Generating Dataloader
    loader =  Loader(dataset)
    scorers = Multi_scoring(loader, Encoder, device, dataset_name=config["dataset"])

    try :
        test_dataset_config = config["testing_dataset"]
    except :
        test_dataset_config = config["dataset"]
    if(test_dataset_config != config["dataset"]):
	    #for using Vox1 as a test set
        with open("data/"+test_dataset_config+".yaml", "r") as ymlfile:
            test_dataset = yaml.full_load(ymlfile)
        test_loader = Loader(test_dataset)
        logging.info("New Test data loaded from : "+test_dataset_config+" Size of test loader {}".format(test_loader.get_dataloader("test").__len__()))
        scorers.change_test(test_loader, test_dataset_config)


    # Initiating Solver
    solver = Solver(loader.get_loaders(), config, dataset, Encoder, scorers = scorers)
    logging.info("solver initialized")

    # Training
    solver.train()

    logging.info("### Training Finished ###")


