import os, argparse, sys, yaml
from torch.nn.functional import threshold
from torch.backends import cudnn
import torch
import sidekit
from loader import Loader
from solver_encoder import Solver
from scoring import Scoring, Multi_scoring
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
    parser.add_argument('--config', type=str, default='configs/500ep_halfresnetAnthony.yaml', help='yaml conf file for the experiment')
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

    with open("data/"+config["dataset"]+".yaml", "r") as ymlfile:
        dataset = yaml.full_load(ymlfile)
    logging.info("Dataset loaded :"+config["dataset"])

    # Loading model encoder
    device = torch.device("cuda")
    checkpoint = torch.load(config["model"]["encoder_dir"], map_location=device)
    speaker_number = checkpoint["speaker_number"]
    model_archi = checkpoint["model_archi"]["model_type"]
    try:
        model_archi = checkpoint["model_archi"]#["model_type"]
        Encoder = sidekit.nnet.xvector.Xtractor(speaker_number, model_archi=model_archi, loss=checkpoint["loss"])
    except:
        model_archi = checkpoint["model_archi"]["model_type"]
        Encoder = sidekit.nnet.xvector.Xtractor(speaker_number, model_archi=model_archi, loss=checkpoint["loss"])

    Encoder.load_state_dict(checkpoint["model_state_dict"])
    Encoder = Encoder.eval().cuda().to(device)
    logging.info("Encoder loaded from : "+config["model"]["encoder_dir"])


     # Generating Dataloader
    loader =  Loader(dataset)
    encoder_name = config["model"]["encoder_dir"].split("/")[-1].strip(".pt")
    scorers = Multi_scoring(loader, Encoder, device, dataset_name=config["dataset"], encoder_name=encoder_name)

    try :
        test_dataset_config = config["testing_dataset"]
    except :
        test_dataset_config = config["dataset"]
    if(True and test_dataset_config != config["dataset"]):
	    #for using Vox1 as a test set
        with open("data/"+test_dataset_config+".yaml", "r") as ymlfile:
            test_dataset = yaml.full_load(ymlfile)
        test_loader = Loader(test_dataset)
        logging.info("New Test data loaded from : "+test_dataset_config+" Size of test loader {}".format(test_loader.get_dataloader("test").__len__()))
        scorers.change_test(test_loader, test_dataset_config)


    # Initiating Solver
    config["model"]["from_loading"] = True
    config["model"]["from_loading"] = True
    solver = Solver(loader.get_loaders(), config, dataset, Encoder, scorers = scorers)
    logging.info("solver initialized")

    #EER of sets
    logging.info("EER on test set : "+str(scorers.back_test_scorer.compute_EER())+" %")
    logging.info("EER on train set : "+str(scorers.val_scorer.compute_EER())+" %")
    logging.info("EER on val set : "+str(scorers.train_scorer.compute_EER())+" %")
    

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

    scorers.EER_post_generator_evaluation("train", Encoder, solver.G )
    scorers.EER_post_generator_evaluation("valid", Encoder, solver.G )
    scorers.EER_post_generator_evaluation("eval", Encoder, solver.G )

    #scorers.compute_TAR("train", Encoder, solver.G, 0.01)
    scorers.compute_TAR("valid", Encoder, solver.G, 0.01)
    scorers.compute_TAR("eval", Encoder, solver.G, 0.01)
    exit()
    scorers.compute_full_TAR(Encoder, solver.G, 0.01)
    scorers.compute_full_TAR(Encoder, solver.G, 0.02)
    scorers.compute_full_TAR(Encoder, solver.G, 0.05)
    scorers.compute_full_TAR(Encoder, solver.G, 0.1)
    logging.info("### Evaluation Finished ###")


