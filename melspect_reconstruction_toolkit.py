import torch
import torch.nn.functional as F
import datetime
import numpy as np
import sidekit
import logging
import logging.config
import matplotlib.pyplot as plt
import os, argparse, sys, yaml
from torch.backends import cudnn
from loader import Loader
from solver_encoder import Solver
from scoring import Scoring, Multi_scoring
from griffin_lim.audio_utilities import save_audio_to_file


class Reconstructor(torch.nn.Module):
    def __init__(self, 
                batch_size=2,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                n_mels=80,
                input_length=128,
                device=None):
        super(Reconstructor, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.input_length = input_length
        self.output_length = input_length*(hop_length-1)
        self.win_input = int(win_length/hop_length)
        self.batch_size = batch_size

        self.conv = torch.nn.Conv2d(1,1,(1,4),stride=1)
        self.FC1 = torch.nn.Linear(self.n_mels,self.hop_length//2)
        self.FC11 = torch.nn.Linear(self.hop_length//2, self.hop_length//2)
        self.FC2 = torch.nn.Linear(self.hop_length//2, self.hop_length)
        mask = np.zeros(shape=(self.input_length, self.input_length, self.win_input))
        for i in range(2, mask.shape[0]-2):
            mask[i,i-2:i+2] = np.eye(self.win_input)
        mask[0], mask[1] = mask[2], mask[2]
        mask[-1], mask[-2] = mask[-3], mask[-3]
        self.splitting_mask = torch.from_numpy(mask).float().unsqueeze(0).repeat(self.batch_size, 1, 1, 1).to(device)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    
    def forward(self,x):
        x_batch = x.repeat(1, self.splitting_mask.shape[1], 1, 1)
        x_batch = torch.matmul(x_batch, self.splitting_mask)
        x_batch = self.relu(self.conv(x_batch.view(self.batch_size*self.input_length,self.n_mels, self.win_input).unsqueeze(1))).squeeze(1).squeeze(2).view(self.batch_size,self.input_length,self.n_mels)
        x_batch = self.relu(self.FC1(x_batch))
        x_batch = self.relu(self.FC11(x_batch))
        x_batch = self.tanh(self.FC2(x_batch))
        return x_batch.view(self.batch_size, x_batch.shape[1]*x_batch.shape[2])[:,:self.output_length]


class Solver_Reconstructor(object):
    def __init__(self,
                config_, 
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                n_mels=80,
                input_length=128):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.input_length = input_length

        EXPE_NAME = config_.split("/")[1].strip(".yaml")
        with open(config_, "r") as ymlfile:
            config = yaml.full_load(ymlfile)
        logging.info("Config loaded from :"+config_)
        config["EXPE_NAME"]=EXPE_NAME

        with open("data/"+config["dataset"]+".yaml", "r") as ymlfile:
            dataset = yaml.full_load(ymlfile)
        logging.info("Dataset loaded :"+config["dataset"])

        # Loading model encoder
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        logging.info("device used : "+ str(self.device))
        checkpoint = torch.load(config["model"]["encoder_dir"], map_location=self.device)
        speaker_number = checkpoint["speaker_number"]
        model_archi = checkpoint["model_archi"]["model_type"]
        logging.info("model archi : {}".format(model_archi))
        self.Encoder = sidekit.nnet.xvector.Xtractor(speaker_number, model_archi=model_archi, loss=checkpoint["loss"])
        self.Encoder.load_state_dict(checkpoint["model_state_dict"])
        self.Encoder = self.Encoder.eval().cuda().to(self.device)
        logging.info("Encoder loaded from : "+config["model"]["encoder_dir"] )


        # Generating Dataloader
        loader =  Loader(dataset)
        scorers = Multi_scoring(loader, self.Encoder, self.device, dataset_name=config["dataset"])

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
        
        self.train_loader = loader.get_dataloader(set_name="train")
        self.val_loader = loader.get_dataloader(set_name="valid")
        self.Rec = Reconstructor(batch_size=64, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.Rec.parameters(), 0.00005)
        self.Rec.eval().to(self.device)
        self.loss = torch.nn.MSELoss()

        logging.info("Solver Initialized !")

    def train(self, epochs=1):
        logging.info("starting of the training")

        for epoch in range(epochs):
            losses = []
            for voices, speakers in self.train_loader:
                self.optimizer.zero_grad()
                voices = voices.squeeze(1).to(self.device)
                mel_spects = self.Encoder(voices,is_eval=True, only_preprocessing=True)
                est_voices = self.Rec(mel_spects.unsqueeze(1))
                #logging.info(est_voices.shape, voices.shape)
                loss = self.loss(est_voices, voices)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.mean().item())
            
            logging.info("epoch {} MSE loss : {}".format(epoch+1, np.mean(losses)))
        logging.info("training finished")

    def test(self):
        for voices, speakers in self.val_loader:
            voices = voices.squeeze(1).to(self.device)
            logging.info(voices.shape)
            save_audio_to_file(voices[0].detach().cpu().numpy(), 16000, outfile='in.wav')
            mel_spects = self.Encoder(voices,is_eval=True, only_preprocessing=True)
            est_voices = self.Rec(mel_spects.unsqueeze(1))
            logging.info(est_voices.shape)
            break
        save_audio_to_file(est_voices[0].detach().cpu().numpy(), 16000, outfile='out.wav')


if __name__=="__main__":
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

    #logging
    log_format = logging.Formatter('[%(asctime)s] \t %(message)s')
    log = logging.getLogger()                                  
    log.setLevel(logging.INFO)  
    LOG_CONFIG = {'version':1,
              'handlers':{'console':{'class':'logging.StreamHandler'},
                          'file':{'class':'logging.FileHandler',
                                  'filename':'logs/reconstruct.log'}},
              'root':{'handlers':('console', 'file'), 
                      'level':'DEBUG'}}
    logging.config.dictConfig(LOG_CONFIG)

    solver = Solver_Reconstructor(args.config)
    solver.train(epochs=25)
    solver.test()

    """Rec = Reconstructor()
    mel_spect = torch.rand((2,1,80,128))
    out = Rec(mel_spect)
    logging.info("output :", out.shape)"""
