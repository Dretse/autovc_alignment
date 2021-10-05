import pandas as pd
import numpy as np

half = 1

Vox1 = pd.read_csv("data/voxceleb_dev_half1.csv")
Vox2 = pd.read_csv("data/voxceleb_dev_half2.csv")
Vox2["gender"] = Vox2["gender"].str.strip(',')
Vox1["gender"] = Vox1["gender"].str.strip(',')
print(Vox2.head())
print(Vox1.head())
print(Vox2.shape, len(set(Vox2["speaker_idx"])), len(Vox2[Vox2["gender"]=='m']), len(Vox2[Vox2["gender"]=='f']))
print(Vox1.shape, len(set(Vox1["speaker_idx"])), len(Vox1[Vox1["gender"]=='m']), len(Vox1[Vox1["gender"]=='f']))

Vox1["speaker_idx"] = Vox1["speaker_idx"]+len(set(Vox2["speaker_idx"]))
#Vox2["gender"] = Vox2["gender"].str.strip(',')
#Vox1["file_id"] = 'vox1_dev_wav/wav/' + Vox1["file_id"].astype(str)
#Vox2["file_id"] = 'vox2_dev_wav/wav/' + Vox2["file_id"].astype(str)

Vox = pd.concat([Vox2, Vox1], ignore_index=True)
print(Vox.head())
print(Vox.shape, len(set(Vox["speaker_idx"])), len(Vox[Vox["gender"]=='m']), len(Vox[Vox["gender"]=='f']) )
print(Vox[Vox["speaker_idx"]==3563])
print("speaker number :",len(set(Vox["speaker_idx"])), np.max(Vox["speaker_idx"]))
Vox.to_csv("data/voxceleb_dev.csv", index = False)

