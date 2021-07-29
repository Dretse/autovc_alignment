import sidekit, torch, os
import numpy as np
from torch.nn.functional import threshold
from torch.utils import data
from fast_eer import eer, eer_threshold
from time import time
from tqdm import tqdm
import logging

class Scoring():
    def __init__(self, dataloader, dataset, device, name="test",dataset_name="VCTK", n_uttrs=1):
        self.emb_dim = 256
        self.dataloader = dataloader
        self.dataset = dataset
        self.device = device
        self.name = name
        self.dataset_name = dataset_name
        self.batch_size=64
        self.len = self.batch_size*self.dataloader.__len__() if name=="train" else len(dataset) 
        self.n_uttrs = n_uttrs # WARNING : 0³ Complexity for this parameter and the number of users

        # definition of paths
        self.users_path = "data/users_"+name+"_"+dataset_name+".npy"
        self.embeddings_path = "data/embeddings_"+name+"_"+dataset_name+".npy"

    def extract_embeddings(self, encoder):
        if(os.path.exists(self.users_path) and os.path.exists(self.embeddings_path)):
            self.Users = np.load(self.users_path).astype(int)
            self.embeddings = np.load(self.embeddings_path)
            #logging.info("Embeddings and users loaded {} {}".format(self.embeddings.shape, self.Users.shape))
        else:
            start = time()
            self.len = min(self.len, 2000)
            embeddings = np.zeros((self.len, self.emb_dim))
            users = np.zeros(self.len)
            with torch.no_grad():
                for batch_idx, data in enumerate(self.dataloader):

                    if self.name!="test":
                        voice, user = data[0].to(self.device), data[1]  
                    else:
                        try:
                            voice, user = data[0].to(self.device), int(data[1][0][1:])
                        except:
                            voice, user = data[0].to(self.device), int(data[1][0][2:])
                    if(len(voice.shape)>2): voice = voice.squeeze(1)
                    with torch.cuda.amp.autocast(enabled=False):
                        pred, embedding = encoder(voice, is_eval=True)

                    if self.name!="test":
                        try:
                            embeddings[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size,:] = embedding.detach().cpu().numpy()
                            users[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size] = user
                        except:
                            embeddings[-embedding.size(0):,:] = embedding.detach().cpu().numpy()
                            users[-user.size(0):] = user.numpy()
                            break
                        
                    else: 
                        try:
                            embeddings[batch_idx,:] = embedding.detach().cpu().numpy()
                            users[batch_idx] = user
                        except: break
                    

            self.embeddings = embeddings
            self.Users = users.astype(int)
            logging.info("{} {} \tEmbeddings extracted : {} s.".format(embeddings.shape, users.shape, int(time()-start)))
            np.save(self.embeddings_path,self.embeddings)
            np.save(self.users_path, self.Users)
        
        self.list_Users = np.array(list(set(self.Users))).astype(int)
        self.mean_embeddings = np.array([ np.mean(self.embeddings[np.argwhere(self.Users==u)[:,0]], axis=0) for u in self.list_Users])
        mask = ((np.outer(self.Users + 1, 1 / (self.Users + 1)) == 1).astype(int) * 2 - 1)
        self.mask = mask[np.tril_indices(mask.shape[0], -1)]

        # definition of gen masks
        if(len(self.list_Users)>150): self.list_Users = self.list_Users[:150]
        n_user = self.n_uttrs*len(self.list_Users)
        #print("début calcul masks", n_user, len(self.list_Users), n_user**4)
        users = np.expand_dims(self.list_Users.repeat(self.n_uttrs),0).repeat(len(self.list_Users),axis=0).reshape((n_user*len(self.list_Users)))
        self.tgt_mask = ((np.outer(users + 1, 1 / (users + 1)) == 1).astype(int) * 2 - 1)
        self.tgt_mask = self.tgt_mask[np.tril_indices(self.tgt_mask.shape[0], -1)]

        users = np.expand_dims(self.list_Users.repeat(1),0).repeat(n_user,axis=0).T.reshape((n_user*len(self.list_Users)))
        self.src_mask = ((np.outer(users + 1, 1 / (users + 1)) == 1).astype(int) * 2 - 1)
        self.src_mask = self.src_mask[np.tril_indices(self.src_mask.shape[0], -1)] 
        #print("masks calculés")
        logging.info("Masks dimensions : "+str(self.mask.shape)+"  "+str(self.tgt_mask.shape)+"  "+str(self.src_mask.shape))
    
    def compute_EER(self):
        if(self.embeddings is None): 
            logging.error("Error : compute first a set of embeddings before computing the EER !")
            exit()
        scores = torch.mm(torch.from_numpy(self.embeddings), torch.from_numpy(self.embeddings).T).numpy()
        scores = scores[np.tril_indices(scores.shape[0], -1)]
        negatives = scores[np.argwhere(self.mask == -1)][:, 0].astype(float)
        positives = scores[np.argwhere(self.mask == 1)][:, 0].astype(float)
        return 100*eer(negatives, positives)
    
    def compute_EER_gen(self, encoder, generator):
        
        embeddings = np.zeros((self.n_uttrs*len(self.list_Users), len(self.list_Users), self.emb_dim))
        #users = np.zeros((len(self.list_Users), len(self.list_Users), 2))
        source_list = np.zeros(len(self.list_Users)).astype(int)
        with torch.no_grad():
            # pour chaque segment d'un dataset
            for batch_idx, data in enumerate(self.dataset):
                voice, speaker = data[0].to(self.device), data[1]
                if(type(speaker)==type("")):
                    try:speaker = int(speaker[1:])
                    except:speaker = int(speaker[2:])
                else: speaker = speaker.item()
                #Si le speaker associé n'a pas encore été vu (ou vu moins de fois que n_uttrs)
                try:
                    src_list_idx = np.argwhere(self.list_Users==speaker)[:,0][0]
                except:
                    #print("error ?", speaker)
                    src_list_idx = None
                    #print(self.list_Users)
                    #exit()

                if(src_list_idx!=None and source_list[src_list_idx]< self.n_uttrs):
                    

                    # Les embeddings moyens de tous les speakers
                    tgt_embeddings = torch.from_numpy(np.array([self.mean_embeddings[np.argwhere(self.list_Users==spk_)[:,0]] for spk_ in set(self.list_Users)])[:,0,:]).to(self.device)
                    # L'embedding de ce speaker répété N fois
                    src_embeddings = torch.from_numpy(self.mean_embeddings[np.argwhere(self.list_Users==speaker)[:,0]]).repeat(len(self.list_Users),1).to(self.device)

                    with torch.cuda.amp.autocast(enabled=False):
                        filter_banks = encoder(voice, is_eval=True, only_preprocessing=True).repeat(len(self.list_Users),1,1)
                        if(filter_banks.shape[2]>128):
                            filter_banks = filter_banks[:,:,:128]
                        #logging.info(str(filter_banks.transpose(1,2).shape)+str(src_embeddings.shape)+str(tgt_embeddings.shape))
                        outputs, outputs_psnt, _ = generator(filter_banks.transpose(1,2).float(), src_embeddings.float(), tgt_embeddings.float())
                        #logging.info(str(outputs.shape))
                        _, embedding = encoder(outputs.squeeze(1).transpose(1,2), is_eval=True, preprocessing=False)
                    #users[batch_idx, idx] = [speaker, tgt_user]
                    #print(speaker*self.n_uttrs + source_list[src_list_idx], source_list[src_list_idx]*len(self.list_Users) , (source_list[src_list_idx]+1)*len(self.list_Users))
                    if(self.n_uttrs==1):embeddings[src_list_idx] = embedding.detach().cpu()
                    else:
                        embeddings[src_list_idx*self.n_uttrs + source_list[src_list_idx]] = embedding.detach().cpu()
                    source_list[src_list_idx]+=1
                if(np.sum(source_list)==self.n_uttrs*len(self.list_Users)):break

        n_user = self.n_uttrs*len(self.list_Users)
        self.embeddings_gen = embeddings
        #Compute tgt scores
        embeddings = self.embeddings_gen.reshape((n_user*len(self.list_Users),self.embeddings_gen.shape[-1]))
        scores = torch.mm(torch.from_numpy(embeddings).to(self.device), torch.from_numpy(embeddings).to(self.device).T).cpu().numpy()
        scores = scores[np.tril_indices(scores.shape[0], -1)]
        negatives = scores[np.argwhere(self.tgt_mask == -1)][:, 0].astype(float)
        positives = scores[np.argwhere(self.tgt_mask == 1)][:, 0].astype(float)
        eer_tgt = 100*eer(negatives, positives)
        logging.info("EER target "+self.name+" : "+str(eer_tgt)+" %")

        #Compute source scores
        scores = torch.mm(torch.from_numpy(embeddings).to(self.device), torch.from_numpy(embeddings).to(self.device).T).cpu().numpy()
        scores = scores[np.tril_indices(scores.shape[0], -1)]
        negatives = scores[np.argwhere(self.src_mask == -1)][:, 0].astype(float)
        positives = scores[np.argwhere(self.src_mask == 1)][:, 0].astype(float)
        eer_src = 100*eer(negatives, positives)
        logging.info("EER source "+self.name+" : "+str(eer_src)+" %")
    
    def compute_TAR(self, threshold_percent=-1):
        if(self.embeddings_gen is None): 
            logging.error("Error : compute first a set of embeddings using the generator before computing the EER !")
            exit()
        
        #compute threshold
        scores = torch.mm(torch.from_numpy(self.embeddings), torch.from_numpy(self.embeddings).T).numpy()
        scores = scores[np.tril_indices(scores.shape[0], -1)]
        negatives = scores[np.argwhere(self.mask == -1)][:, 0].astype(float)
        positives = scores[np.argwhere(self.mask == 1)][:, 0].astype(float)
        
        if(threshold_percent==-1): 
            threshold = eer_threshold(negatives, positives)
        elif(threshold_percent==0):
            threshold = np.max(negatives)
        else: 
            threshold = np.sort(negatives)[int((1-threshold_percent)*len(negatives))]
        #logging.info(threshold)
        #compute scores
        tgt_embeddings = self.embeddings_gen.reshape((self.n_uttrs*len(self.list_Users)*len(self.list_Users),self.embeddings_gen.shape[-1]))
        users = np.expand_dims(self.list_Users.repeat(self.n_uttrs),0).repeat(len(self.list_Users),axis=0).reshape((self.n_uttrs*len(self.list_Users)*len(self.list_Users)))
        tar_mask = ((np.outer(self.Users + 1, 1 / (users + 1)) == 1).astype(int))

        #logging.info(str(tar_mask.shape))
        #print(tgt_embeddings.shape, self.embeddings.shape)
        scores = torch.mm(torch.from_numpy(self.embeddings).to(self.device), torch.from_numpy(tgt_embeddings).to(self.device).T).cpu().numpy()
        #scores = scores.flatten()[np.argwhere(tar_mask == 1)[:, 1]].astype(float)
        scores = scores[tar_mask==1].astype(float)
        #logging.info(str(scores.shape))
        return 100*np.sum(scores>threshold)/len(scores)


class Multi_scoring():
    def __init__(self, loader, Encoder, device, name="test",dataset_name="VCTK", print_eers=False):
        logging.info("Initiating Scorers for "+dataset_name)
        self.Encoder = Encoder
        self.device = device
        self.dataset_name = dataset_name
        #Generation Scorers
        self.train_scorer = Scoring(loader.get_dataloader("train"), loader.get_dataset("train"), device, name="train",dataset_name=dataset_name)
        self.train_scorer.extract_embeddings(Encoder)
        if(print_eers):logging.info("EER on train set : "+str(self.train_scorer.compute_EER())+" %")


        self.val_scorer = Scoring(loader.get_dataloader("valid"), loader.get_dataset("valid"), device, name="valid", dataset_name=dataset_name)
        self.val_scorer.extract_embeddings(Encoder)
        if(print_eers):logging.info("EER on val set : "+str(self.val_scorer.compute_EER())+" %")

        #Initiating scoring modules
        self.test_scorer = Scoring(loader.get_dataloader("test"), loader.get_dataset("test"), device, name="test", dataset_name=dataset_name)
        self.test_scorer.extract_embeddings(Encoder)
        self.test_dataset_name = dataset_name

        self.back_test_scorer = Scoring(loader.get_dataloader("test"), loader.get_dataset("test"), device, name="test", dataset_name=dataset_name)
        self.back_test_scorer.extract_embeddings(Encoder)
        if(print_eers):logging.info("EER on test set : "+str(self.test_scorer.compute_EER())+" %")

    def get_scorers(self):
        return (self.train_scorer, self.val_scorer, self.test_scorer)

    def change_test(self,loader, dataset_name):
        self.test_scorer = Scoring(loader.get_dataloader("test"), loader.get_dataset("test"), self.device, name="test", dataset_name=dataset_name)
        self.test_scorer.extract_embeddings(self.Encoder)
        self.test_dataset_name = dataset_name
        logging.info("EER on test set : "+str(self.test_scorer.compute_EER())+" %")
        self.generate_masks()
        logging.info("New test set added")


    def get_dataset(self, dataset_name):
        return self.get_scorer(dataset_name).dataset

    def get_dataloader(self, dataset_name):
        return self.get_scorer(dataset_name).dataloader
    
    def get_scorer(self, dataset_name):
        if(dataset_name=="train"):
            return self.train_scorer
        elif(dataset_name=="valid"):
            return self.val_scorer
        elif(dataset_name=="eval"):
            return self.test_scorer
        else:
            logging.error("ERROR : scorer furnished not in {train, valid, eval}. laoder furnished :"+dataset_name)
            exit()

    def EER_post_generator_evaluation(self,loader_used, encoder, generator):
        scorer = self.get_scorer(loader_used)
        if(loader_used =="eval" and self.test_dataset_name != self.dataset_name):
            self.compute_EER_gen(encoder, generator, self.back_test_scorer, self.test_scorer)
        else:
            self.get_scorer(loader_used).compute_EER_gen(encoder, generator)

    def generate_masks(self):
        tgt_users, src_users = self.test_scorer.list_Users, self.back_test_scorer.list_Users
        if(len(tgt_users)>150): tgt_users = tgt_users[:150]
        if(len(src_users)>150): src_users = src_users[:150]
        n_user_src, n_user_tgt = len(src_users), len(tgt_users)

        #print("début calcul masks", n_user, len(self.list_Users), n_user**4)
        users = np.expand_dims(tgt_users,0).repeat(n_user_src,axis=0).reshape((n_user_tgt*n_user_src))
        self.tgt_mask = ((np.outer(users + 1, 1 / (users + 1)) == 1).astype(int) * 2 - 1)
        self.tgt_mask = self.tgt_mask[np.tril_indices(self.tgt_mask.shape[0], -1)]

        users = np.expand_dims(src_users,0).repeat(n_user_tgt,axis=0).T.reshape((n_user_tgt*n_user_src))
        self.src_mask = ((np.outer(users + 1, 1 / (users + 1)) == 1).astype(int) * 2 - 1)
        self.src_mask = self.src_mask[np.tril_indices(self.src_mask.shape[0], -1)] 
        #print("masks calculés")
        logging.info("Masks dimensions : "+str(self.tgt_mask.shape)+"  "+str(self.src_mask.shape))
    
    def compute_EER_gen(self, encoder, generator, source_dataset, target_dataset):
        embeddings = np.zeros((len(source_dataset.list_Users), len(target_dataset.list_Users), target_dataset.emb_dim))
        print(len(source_dataset.list_Users), len(target_dataset.list_Users))
        assert(target_dataset.emb_dim==source_dataset.emb_dim)
        source_list = np.zeros(len(source_dataset.list_Users)).astype(int)
        with torch.no_grad():
            # pour chaque segment d'un dataset
            for batch_idx, data in enumerate(source_dataset.dataset):
                voice, speaker_src = data[0].to(self.device), data[1]
                if(type(speaker_src)==type("")):
                    try:speaker_src = int(speaker_src[1:])
                    except:speaker_src = int(speaker_src[2:])
                else: speaker_src = speaker_src.item()
                #Si le speaker_src associé n'a pas encore été vu
                try:
                    src_list_idx = np.argwhere(source_dataset.list_Users==speaker_src)[:,0][0]
                except:
                    #print("error ?", speaker_src)
                    src_list_idx = None
                    #print(self.list_Users)
                    #exit()

                if(src_list_idx!=None and source_list[src_list_idx]==0):
                    

                    # Les embeddings moyens de tous les speakers
                    tgt_embeddings = torch.from_numpy(np.array([target_dataset.mean_embeddings[np.argwhere(target_dataset.list_Users==spk_)[:,0]] for spk_ in set(target_dataset.list_Users)])[:,0,:]).to(self.device)
                    # L'embedding de ce speaker_src répété N fois
                    src_embeddings = torch.from_numpy(source_dataset.mean_embeddings[np.argwhere(source_dataset.list_Users==speaker_src)[:,0]]).repeat(len(target_dataset.list_Users),1).to(self.device)
                    #print(tgt_embeddings.shape, src_embeddings.shape)
                    with torch.cuda.amp.autocast(enabled=False):
                        filter_banks = encoder(voice, is_eval=True, only_preprocessing=True).repeat(len(target_dataset.list_Users),1,1)
                        if(filter_banks.shape[2]>128):
                            filter_banks = filter_banks[:,:,:128]
                        #logging.info(str(filter_banks.transpose(1,2).shape)+str(src_embeddings.shape)+str(tgt_embeddings.shape))
                        outputs, outputs_psnt, _ = generator(filter_banks.transpose(1,2).float(), src_embeddings.float(), tgt_embeddings.float())
                        #logging.info(str(outputs.shape))
                        _, embedding = encoder(outputs.squeeze(1).transpose(1,2), is_eval=True, preprocessing=False)
                    #users[batch_idx, idx] = [speaker_src, tgt_user]
                    embeddings[src_list_idx] = embedding.detach().cpu()
                    source_list[src_list_idx]+=1
                if(np.sum(source_list)==len(source_dataset.list_Users)):break

        #n_user = len(self.list_Users)
        #print(embeddings.shape)
        #Compute tgt scores
        embeddings = embeddings.reshape((len(target_dataset.list_Users)*len(source_dataset.list_Users),embeddings.shape[-1]))
        print(embeddings.shape)
        scores = torch.mm(torch.from_numpy(embeddings).to(self.device), torch.from_numpy(embeddings).to(self.device).T).cpu().numpy()
        scores = scores[np.tril_indices(scores.shape[0], -1)]
        print(scores.shape, self.tgt_mask.shape, target_dataset.src_mask.shape, source_dataset.src_mask.shape, source_dataset.tgt_mask.shape)
        negatives = scores[np.argwhere(self.tgt_mask == -1)][:, 0].astype(float)
        positives = scores[np.argwhere(self.tgt_mask == 1)][:, 0].astype(float)
        eer_tgt = 100*eer(negatives, positives)
        logging.info("EER target "+target_dataset.name+" : "+str(eer_tgt)+" %")

        #Compute source scores
        scores = torch.mm(torch.from_numpy(embeddings).to(self.device), torch.from_numpy(embeddings).to(self.device).T).cpu().numpy()
        scores = scores[np.tril_indices(scores.shape[0], -1)]
        negatives = scores[np.argwhere(self.src_mask == -1)][:, 0].astype(float)
        positives = scores[np.argwhere(self.src_mask == 1)][:, 0].astype(float)
        eer_src = 100*eer(negatives, positives)
        logging.info("EER source "+source_dataset.name+" : "+str(eer_src)+" %")