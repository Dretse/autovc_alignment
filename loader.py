import torch, sidekit, yaml, pandas
from torch.utils.data import DataLoader
import numpy as np
import logging

class Loader():
    def __init__(self, data_config):
        self.conf = data_config
        df = pandas.read_csv("data/"+data_config["dataset"]+".csv") 
        #selection mic1 mic2 for VCTK
        if(data_config["dataset"]=="VCTK"):
            logging.info("only use mic1 sequences")
            df = df[[int(i[-1])==1 for i in df['file_id']]]
            """elif(not data_config["mic1"]):
                logging.info("only use mic2 sequences")
                df = df[[int(i[-1])==2 for i in df['file_id']]]"""
        
        self.speaker_number = len(set(df["speaker_idx"]))
        # split train test
        train_df = df[            df["speaker_idx"]>=data_config["train"]["users"][0]]
        train_df = train_df[train_df["speaker_idx"]<=data_config["train"]["users"][1]]
        test_df  =   df[          df["speaker_idx"]>=data_config["test"]["users"][0]]
        test_df  =   test_df[test_df["speaker_idx"]<=data_config["test"]["users"][1]]  
        train_df = train_df[train_df['duration']>=data_config["train"]["duration"]]
        # split train valid     
        idx = np.arange(len(train_df))
        np.random.shuffle(idx)
        train_df, val_df = train_df.iloc[idx[:int(len(idx)*0.9)]], train_df.iloc[idx[int(len(idx)*0.9):]]

        self.val_df = val_df.reset_index(drop=True)
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df[test_df['duration']>=data_config["test"]["duration"]].reset_index(drop=True)

        #create IdMapSets
        self.test_set = self.get_idmapset(self.test_df)
        self.train_set = self.get_sideset(self.train_df, 'train')
        self.val_set = self.get_sideset(self.val_df, 'valid')

    def get_idmapset(self, df):
        idmap = sidekit.IdMap()
        idmap.leftids = np.array(df['speaker_id'])
        idmap.rightids = np.array(df['file_id'])
        idmap.start = np.empty_like(idmap.leftids, dtype=None)
        idmap.stop = np.empty_like(idmap.leftids, dtype=None)
        idmap.validate()

        testing_set = sidekit.nnet.xsets.IdMapSet(idmap_name=idmap,
                            data_path=self.conf["data_path"],
                            file_extension=self.conf["data_file_extension"][1:],
                            sliding_window=self.conf["sliding_window"],
                            window_len=self.conf["window_len"],
                            window_shift=self.conf["window_shift"],
                            sample_rate=self.conf["sample_rate_target"],
                            min_duration=2.1
                            )

        return testing_set
    
    def get_sideset(self, df, set_name="train"):
        #charge the SideSet
        training_set = sidekit.nnet.xsets.SideSet(self.conf,
                           set_type="train",
                           chunk_per_segment=self.conf[set_name]["chunk_per_segment"],
                           overlap=self.conf[set_name]["overlap"],
                           dataset_df=df,
                           output_format="pytorch",
                           min_duration=self.conf["train"]["duration"]
                           )
        return training_set
    
    def get_dataloader(self, set_name="None"):
        dataset = self.get_dataset(set_name)

        return DataLoader(dataset, 
                num_workers=self.conf["num_workers"], 
                batch_size=self.conf[set_name]["batch_size"],
                shuffle = self.conf[set_name]["shuffle"],
                sampler= self.get_sidesampler() if set_name=="train" else None)
        
    def get_sidesampler(self):
        # Only for the train set
        #logging.info(str(set(self.train_df['speaker_idx'])))
        side_sampler = sidekit.nnet.xsets.SideSampler(data_source=np.array(self.train_df['speaker_idx']),
                                   spk_count=len(set(self.train_df["speaker_idx"])),
                                   examples_per_speaker=self.conf["train"]["examples_per_speaker"],#2
                                   samples_per_speaker=self.conf["train"]["samples_per_speaker"],#32
                                   batch_size=self.conf["train"]["batch_size"],#64
                                   )
        return side_sampler
    
    def get_dataset(self, set_name="test"):
        if(set_name=="test"):
            return self.test_set
        elif(set_name=="valid"):
            return self.val_set
        elif(set_name=="train"):
            return self.train_set
        else:
            logging.error("Error : set_name is "+set_name+" not in {test, train, valid} options")
            exit()
    
    def get_loaders(self):
        return (self.get_dataloader("train"), self.get_dataloader("valid"), self.get_dataloader("test"))
