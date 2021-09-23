import pandas as pd
import numpy as np

Vox2 = pd.read_csv("data/voxceleb1_dev.csv")
print(Vox2.head())
print(Vox2.shape, len(set(Vox2["speaker_idx"])))
Vox2_0 = Vox2[Vox2["speaker_idx"]%2==0].copy()
Vox2_1 = Vox2[Vox2["speaker_idx"]%2!=0].copy()
Vox2_0 = Vox2_0.reindex()
Vox2_1 = Vox2_1.reset_index(inplace=False).reindex()
Vox2_1.pop("index")
Vox2_0["speaker_idx"] = Vox2_0["speaker_idx"]//2
Vox2_1["speaker_idx"] = Vox2_1["speaker_idx"]//2


print("Vox2 part 1 : shape : {}, males : {}, females :{}, speakers : {}".format(Vox2_0.shape, len(Vox2_0[Vox2_0["gender"]=="m"]), len(Vox2_0[Vox2_0["gender"]=="f"]), len(set(Vox2_0["speaker_idx"]))))
print("Vox2 part 2 : shape : {}, males : {}, females :{}, speakers : {}".format(Vox2_1.shape, len(Vox2_1[Vox2_1["gender"]=="m"]), len(Vox2_1[Vox2_1["gender"]=="f"]), len(set(Vox2_0["speaker_idx"]))))
print(Vox2_0.head())
print(Vox2_1.head())


Vox2_0.to_csv("data/voxceleb1_dev_half1.csv", index = False)
Vox2_1.to_csv("data/voxceleb1_dev_half2.csv", index = False)

