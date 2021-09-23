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
from torchaudio.transforms import GriffinLim, InverseMelScale, Spectrogram
import torchaudio
import matplotlib.pyplot as plt

class GL_rec(torch.nn.Module):
    def __init__(self, 
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                n_mels=80,
                min_freq=90,
                max_freq=7600,
                sample_rate=16000,
                device=None):
        super(GL_rec, self).__init__()
        self.n_fft=n_fft
        self.Mel_spect_inverse = InverseMelScale(n_stft=n_fft,
                                            n_mels=n_mels,
                                            sample_rate=sample_rate,
                                            f_min=min_freq,
                                            f_max=max_freq)
        self.Griffin_Lim = GriffinLim(n_fft=n_fft,
                                    win_length=win_length,
                                    hop_length=hop_length)
        
        self.device = device
    
    def forward(self, mel_spect):
        #mel_spect = torch.exp(mel_spect)
        spect = self.Mel_spect_inverse(mel_spect)
        print(spect.shape)
        #print_spect(spect.cpu().detach().numpy(), "reconstruct")
        if(False):
            nspect = torch.zeros((spect.size(0)+1, spect.size(1))).to(self.device)
            nspect[:-1] += spect
            nspect[1:] += spect
            spect = nspect[[i for i in range(nspect.size(0)) if i%2==0], :]
        else:
            spect = spect[:self.n_fft//2 +1,:]
        print(spect.shape)
        print_spect(spect.cpu().detach().numpy(), "reconstruct_reduct")
        wave = self.Griffin_Lim(spect)
        print(wave.shape)

        #wave = torch.from_numpy(np.array([wave[:i+1].sum().item() for i in range(len(wave))])).float()/0.03
        print(wave.shape)
        return wave


def print_spect(spect, filename):
    fig, axe = plt.subplots(nrows=1, ncols=1)
    axe.imshow(spect)
    axe.set_title("Spectrum "+filename)
    plt.savefig("graphs/griffinlim/"+filename+".png")

def print_wav(wave, filename):
    print(wave.shape)
    fig, axe = plt.subplots(nrows=1, ncols=1)
    axe.plot(np.arange(len(wave)), wave)
    axe.set_title("Waveform "+filename)
    plt.savefig("graphs/griffinlim/"+filename+".png")
"""
if __name__ == "__main__":
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

    with open("data/"+config["dataset"]+".yaml", "r") as ymlfile:
        dataset = yaml.full_load(ymlfile)
    logging.info("Dataset loaded :"+config["dataset"])

    # Loading model encoder
    device = torch.device("cuda")
    checkpoint = torch.load(config["model"]["encoder_dir"], map_location=device)
    speaker_number = checkpoint["speaker_number"]
    model_archi = checkpoint["model_archi"]["model_type"]
    logging.info("model archi : {}".format(model_archi))
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

    Convert = GL_rec(device=device).to(device)
    #Getting a spectrogramm to reconstruct
    for voices, speakers in loader.get_dataloader(set_name="train"):
        print(voices.shape)
        print_wav(voices[0,0].cpu().detach().numpy(), "input")
        torchaudio.save('in.wav', voices.cpu().detach()[0], 16000)
        print(voices.squeeze(1)[0].shape, voices.squeeze(1)[0].min(), voices.squeeze(1)[0].max())
        #mel_spect = Encoder(voices.squeeze(1).to(device),is_eval=True, only_preprocessing=True)[0]
        mel_spect = Spectrogram(
                                n_fft=1024,
                                win_length=1024,
                                hop_length=256,
                            )(voices[0])

        print_spect(mel_spect.cpu().detach().numpy(), "melspect")
        #waveform = Convert(mel_spect).cpu().detach()
        waveform = GriffinLim(
                            n_fft=1024,
                            win_length=1024,
                            hop_length=256
                            )(mel_spect)
        print_wav(waveform.cpu().detach().numpy(), "output")
        torchaudio.save('out.wav', waveform.unsqueeze(0), 16000)
        exit()"""

if(True):
    with open('configs/1ep.yaml', "r") as ymlfile:
        config = yaml.full_load(ymlfile)
    device = torch.device("cuda")
    checkpoint = torch.load(config["model"]["encoder_dir"], map_location=device)
    speaker_number = checkpoint["speaker_number"]
    model_archi = checkpoint["model_archi"]["model_type"]
    logging.info("model archi : {}".format(model_archi))
    Encoder = sidekit.nnet.xvector.Xtractor(speaker_number, model_archi=model_archi, loss=checkpoint["loss"])
    Encoder.load_state_dict(checkpoint["model_state_dict"])
    Encoder = Encoder.eval().cuda().to(device)
    logging.info("Encoder loaded from : "+config["model"]["encoder_dir"] )


voice, _ = torchaudio.load('in.wav', 16000)
voice = voice.to(device)
print(voice.shape)
print_wav(voice[0].cpu().detach().numpy(), "input")
#torchaudio.save('in.wav', voice.cpu().detach(), 16000)
mel_spect = Encoder(voice.squeeze(1).to(device),is_eval=True, only_preprocessing=True)[0]
"""mel_spect = Spectrogram(
                        n_fft=1024,
                        win_length=1024,
                        hop_length=256,
                    )(voice)[0]"""

print(mel_spect.shape)
print_spect(mel_spect.cpu().detach().numpy(), "melspect")


Convert = GL_rec(device=device).to(device)
waveform = Convert(mel_spect).cpu().detach()

print_wav(waveform.cpu().detach().numpy(), "output")
torchaudio.save('out.wav', waveform.unsqueeze(0), 16000)
exit()