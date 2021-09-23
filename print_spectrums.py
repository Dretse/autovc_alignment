import os, argparse, sys, yaml
from torch.nn.functional import embedding, threshold
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
import matplotlib.pyplot as plt

def print_spectrum(spectrums, post_spectrums, filename="graphs/spectrums/noname"):
    if(len(spectrums.shape)==3):
        fig, axes = plt.subplots(nrows=spectrums.shape[0], ncols=2)
        for row in range(spectrums.shape[0]):
            axe_1, axe_2 = axes[row]

            axe_1.imshow(spectrums[row].T)
            axe_1.set_title("Spectrum before reconstruction")
            axe_2.imshow(post_spectrums[row].T)
            axe_2.set_title("Spectrum after reconstruction")
    
    else:
        fig, (axe_1, axe_2) = plt.subplots(nrows=1, ncols=2)

        axe_1.imshow(spectrums.T)
        axe_1.set_title("Spectrum before reconstruction")
        axe_2.imshow(post_spectrums.T)
        axe_2.set_title("Spectrum after reconstruction")
    
    fig.savefig(filename+".png")



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
    parser.add_argument('--config', type=str, default='500ep_neck16_emb10', help='yaml conf file for the experiment')
    parser.add_argument('--n_exp', type=int, default=0, help='number of the epoch')
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

    EXPE_NAME = args.config
    with open('configs/'+args.config+'.yaml', "r") as ymlfile:
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
    config["data"]["dataset_yaml"]="data/VoxCeleb1_DS_5s_clean.yaml"
    config["data"]["file_extention"]="wav"
    config["data"]["train"]["users"]=[0,1089]
    config["data"]["val"]["users"]=[0,1089]
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

    
    batch_size = 3
    for data in val_loader:
        voices, speakers = data[0][:batch_size], data[1][:batch_size]
        break
    #print(voices.shape)
    spectrums = solver.C(voices.squeeze(1).to(solver.device), is_eval=True, only_preprocessing=True).transpose(1,2)
    _, embeddings = solver.C(voices.squeeze(1).to(solver.device), is_eval=True)
    post_spectrums, _, _ = solver.G(spectrums,embeddings, embeddings)
    logging.info(str(batch_size)+" spectrums computed (copy synthesis here)")
    if(not os.path.exists('graphs/spectrums/'+args.config)):
        os.makedirs('graphs/spectrums/'+args.config)
    filename = 'graphs/spectrums/'+args.config+'/'+str(args.n_exp)
    print_spectrum(spectrums.cpu().detach().numpy(), post_spectrums.squeeze(1).cpu().detach().numpy(), filename=filename)

    logging.info("### Plotting Finished ###")


