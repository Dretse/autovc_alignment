import pandas as pd
import numpy as np
import os, torchaudio, tqdm


Vox1 = pd.read_csv("data/voxceleb1_dev_half1.csv")
print(Vox1.head())

set_name = "dev-clean"
root_file = "/ssd/data/LibriSpeech/"+set_name
infos = pd.read_csv("/ssd/data/LibriSpeech/SPEAKERS.TXT", sep='|')
infos.columns = infos.columns.str.replace(' ', '')
print(infos.head())

Ids = []
durations = []
for root, dirs, files in os.walk(root_file):
    if(len(root.split("/"))==5):
        for speaker in dirs:
            for path, _, files in os.walk(os.path.join(root, speaker)):
                for file in files:
                    if(file.split(".")[-1]=="flac"):
                        Ids.append([speaker, os.path.join(path[len(root_file)+1:], file[:-5])])
                        waveform, sample_rate = torchaudio.load(os.path.join(path, file))
                        durations.append(len(waveform[0])/sample_rate)

Ids = np.array(Ids).T
print(Ids.shape)
data = {"speaker_id":Ids[0].astype(int), "file_id":Ids[1]}
data["database"] = np.array([set_name]).repeat(len(Ids[0]))
data["start"] = np.array([0]).repeat(len(Ids[0]))
data["duration"] = np.array(durations)
id_to_idx = sorted(list(set(Ids[0].astype(int))))
id_to_idx = { ids : idx for idx,ids in enumerate(id_to_idx)}
data["speaker_idx"] = np.array([id_to_idx[i] for i in data["speaker_id"]])
data["gender"] = np.array([infos[infos["ID"]==ids]["SEX"].item() for ids in data["speaker_id"]])
dataframe = pd.DataFrame(data)
print(dataframe.head())

dataframe.to_csv("data/LibriSpeech_"+set_name+".csv", index = False)

