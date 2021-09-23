from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import numpy as np
import sidekit
import os
import logging
import matplotlib.pyplot as plt

class Solver(object):

    def __init__(self, loader, config, data_config, encoder_model, scorers=(None,None,None), charge_iteration=0):
        """Initialize configurations."""

        # Data loader.
        self.train_loader, self.val_loader, self.test_loader = loader

        # Model configurations.
        self.len_crop = config["training"]["len_crop"]
        self.lambda_cd = config["model"]["lambda_cd"]
        try:   self.lambda_emb = config["model"]["lambda_emb"]
        except: self.lambda_emb = 10
        self.dim_neck = config["model"]["dim_neck"]
        self.dim_emb = config["model"]["dim_emb"]
        self.dim_pre = config["model"]["dim_pre"]
        self.freq = config["model"]["freq"]

        # Training configurations.
        self.batch_size = data_config["train"]["batch_size"]
        self.num_iters = config["training"]["num_epoch"]
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        logging.info("device used : "+ str(self.device))
        self.log_step = config["training"]["log_step"]
        try : self.use_mean_emb = config["training"]["use_mean_embs"]
        except : self.use_mean_emb = False

        try :
            self.clamp = config["training"]["clamping"]
        except: self.clamp = 0

        try : self.weight_decay = config["training"]["weight_decay"]
        except : self.weight_decay = 0
        

        #loading encoder model
        self.C = encoder_model.requires_grad_(False)


        # Build the model and tensorboard.
        self.build_model()

        self.use_loss_cd = config["training"]["epoch_before_cd"]
        self.use_loss_emb = config["training"]["epoch_before_emb"]
        try :
            self.use_loss_cs = config["training"]["epoch_before_cs"]
        except:
            self.use_loss_cs = 0
        if(self.use_loss_cd<0): self.use_loss_cd  = self.num_iters+1
        if(self.use_loss_emb<0):self.use_loss_emb = self.num_iters+1
        if(self.use_loss_cs<0): self.use_loss_cs  = self.num_iters+1
        try:
            self.use_zero_src_tgt = config["training"]["use_zero_src_tgt"]
            if(self.use_zero_src_tgt): logging.info("Using zero emb instead of target and source embs")
            self.use_rand_src_tgt = config["training"]["use_rand_src_tgt"]
            if(self.use_rand_src_tgt): logging.info("Using random emb instead of target and source embs")
            self.use_true_rand_src_tgt = config["training"]["use_true_rand_src_tgt"]
            if(self.use_true_rand_src_tgt): logging.info("Using truly random emb instead of target and source embs")
        except:
            self.use_zero_src_tgt = False
            self.use_rand_src_tgt = False
            self.use_true_rand_src_tgt = False
        try:
            self.use_zero_like_bottleneck =  config["training"]["use_zero_like_bottleneck"]
        except:
            self.use_zero_like_bottleneck = False

        #Saving directory
        self.savedir = config["training"]["savedir"]+config["EXPE_NAME"]
        self.save_model = config["model"]["save_model"]
        if not os.path.exists(os.path.join(self.savedir)):
            os.makedirs(os.path.join(self.savedir))

        if(not os.path.exists('graphs/spectrums/'+config["EXPE_NAME"])):
            os.makedirs('graphs/spectrums/'+config["EXPE_NAME"])
        self.spectrums_dir = 'graphs/spectrums/'+config["EXPE_NAME"]+'/'+config["EXPE_NAME"]
        #loading directory
        if(config["model"]["from_loading"]):
            try:
                self.loaddir = os.path.join(config["training"]["savedir"],config["model"]["load_dir"], config["model"]["load_dir"])+".ckpt"
                logging.info("loading model from dir :"+self.loaddir)
                if(charge_iteration!=0): self.loaddir = self.loaddir[:-5]+"_"+str(charge_iteration)+".ckpt"
                self.load(self.loaddir)
                
            except:
                logging.error("No loading dir found. Using "+str(self.loaddir)+" by default")
                self.loaddir = os.path.join(self.savedir, config["EXPE_NAME"]+".ckpt")
                if(charge_iteration!=0): self.loaddir = self.loaddir[:-5]+"_"+str(charge_iteration)+".ckpt"
                self.load(self.loaddir)
                

            
            logging.info("Model loaded successfully")
        
        self.savedir = os.path.join(self.savedir, config["EXPE_NAME"]+".ckpt")


        # Scorers importation :
        self.scorers = scorers
        self.train_scorer, self.val_scorer, self.test_scorer = scorers.get_scorers()

        #self.test_zeros()
            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001, weight_decay=self.weight_decay)
        """self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, mode='min', 
                factor=0.5, patience=1000, threshold=0.0001, verbose=True)"""
        self.G.eval().to(self.device)
        
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    def do_iteration(self, x_real, emb_org, train=True, cd=False, emb=False, cs=True):
        self.G = self.G.train()

        # Identity mapping loss
        if(cs):
            if(self.use_zero_src_tgt):        
                x_identic, x_identic_psnt, code_real = self.G(x_real, torch.zeros_like(emb_org), torch.zeros_like(emb_org))
            elif(self.use_true_rand_src_tgt): 
                x_identic, x_identic_psnt, code_real = self.G(x_real, torch.rand_like( emb_org), torch.rand_like( emb_org))
            elif(self.use_rand_src_tgt):
                rand_emb = torch.rand_like( emb_org)
                x_identic, x_identic_psnt, code_real = self.G(x_real,                  rand_emb,                 rand_emb )
            elif(self.use_zero_like_bottleneck):
                x_identic, x_identic_psnt, code_real = self.G(torch.zeros_like(x_real), torch.zeros_like(emb_org), emb_org)
            else:                       
                x_identic, x_identic_psnt, code_real = self.G(x_real,                  emb_org ,                  emb_org )

            x_real = x_real.unsqueeze(1)
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   

        
            if(cd):
                # Code semantic loss.
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)
            else:
                g_loss_cd = torch.zeros(1).to(self.device)

        if(emb):
            # cross Embedding reconstruction loss
            size = x_real.size(0)//2
            x_a, x_b = (x_real[:size], x_real[-size:]) if self.use_zero_like_bottleneck else (x_real[:size,0], x_real[-size:,0])
            emb_a,emb_b = (emb_org[:size], emb_org[-size:]) if self.batch_size!=2 else emb_org.unsqueeze(1)
            #print(x_real.shape, x_a.shape, emb_org.shape, emb_a.shape)
            if(self.use_zero_like_bottleneck):
                _ , x_identic_ab, _ = self.G(torch.zeros_like(x_a), torch.zeros_like(emb_a), emb_b)
                _ , x_identic_ba, _ = self.G(torch.zeros_like(x_b), torch.zeros_like(emb_b), emb_a)
            else:
                _ , x_identic_ab, _ = self.G(x_a, emb_a, emb_b)
                _ , x_identic_ba, _ = self.G(x_b, emb_b, emb_a)

            x_identic_ab = x_identic_ab.squeeze(1)
            x_identic_ba = x_identic_ba.squeeze(1)
            #logging.info(x_identic_psnt_ab.shape, x_identic_psnt_ba.shape)

            pred_ab, emb_ab = self.C(torch.swapaxes(x_identic_ab,1,2), preprocessing=False, is_eval=True)
            pred_ba, emb_ba = self.C(torch.swapaxes(x_identic_ba,1,2), preprocessing=False, is_eval=True)
            g_loss_emb = ( F.mse_loss(emb_b, emb_ab) + F.mse_loss(emb_a, emb_ba) ) /2
        else:
            g_loss_emb = torch.zeros(1).to(self.device)


        # Backward and optimize.
        if(train):
            g_loss =  self.lambda_emb*g_loss_emb
            if(cs): g_loss += self.lambda_cd * g_loss_cd + 0.5*g_loss_id + g_loss_id_psnt
            self.reset_grad()
            g_loss.backward()
            if(self.clamp!=0):torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.clamp)
            self.g_optimizer.step()
        #self.scheduler.step(g_loss)

        # Logging. 
        loss = {}
        if(cs):
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
        loss['G/loss_emb'] = g_loss_emb.item()
        return loss
                


    def train(self):
        # Set data loader.
        torch.cuda.empty_cache()
        data_loader = self.train_loader
        val_loader = self.val_loader
        test_loader = iter(self.test_loader)
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt']
        
        # EER computation
        t = time.time()
        self.scorers.EER_post_generator_evaluation("valid", self.C, self.G)
        best_eer_tgt, _ = self.scorers.EER_post_generator_evaluation("eval", self.C, self.G)
        logging.info("time taken for eers computation : "+str(int(time.time() - t)))
        # Start training.
        logging.info('Start training...')
        start_time = time.time()
        for epoch in range(self.num_iters):

            for data in data_loader:
                # =================================================================================== #
                #                               1. Import the data                                    #
                # =================================================================================== #
                voice, speaker = data
                if(self.use_mean_emb): emb_org = torch.from_numpy(self.train_scorer.mean_embeddings[speaker]).float().to(self.device)
                else: _, emb_org = self.C(voice.squeeze(1).to(self.device),is_eval=True)#Was False
                x_real = self.C(voice.squeeze(1).to(self.device),is_eval=True, only_preprocessing=True).transpose(1,2)#Was False
                x_real = self.random_crop(x_real)
                    
                #logging.info(x_real.shape, emb_org.shape)
                #try:    
                loss = self.do_iteration(x_real, emb_org, train=True, cd=epoch>=self.use_loss_cd, emb=epoch>=self.use_loss_emb, cs=epoch>=self.use_loss_cs)
                """except:
                    logging.error("Error happened during Training iteration number "+str(epoch))
                    exit()"""
                keys = list()
                if(epoch>=self.use_loss_cs): 
                    keys.append('G/loss_id')
                    keys.append('G/loss_id_psnt')
                    if(epoch>=self.use_loss_cd): keys.append('G/loss_cd')
                if(epoch>=self.use_loss_emb): keys.append('G/loss_emb')


            # Print out training information.
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Epoch [{}/{}]\t".format(et, epoch+1, self.num_iters)
            for tag in keys:
                log += ", {}: {:.4f}".format(tag, loss[tag])
            logging.info(str(log))

            # EER computation
            t = time.time()
            self.scorers.EER_post_generator_evaluation("valid", self.C, self.G)
            eer_tgt, _ = self.scorers.EER_post_generator_evaluation("eval", self.C, self.G)
            #self.scorers.EER_post_generator_evaluation("train", self.C, self.G)
            logging.info("time taken for eers computation : "+str(int(time.time() - t)))

            if(True): #Validation Loss
                for val_data in val_loader:
                    voice, speaker = val_data
                    if(self.use_mean_emb): emb_org = torch.from_numpy(self.val_scorer.mean_embeddings[speaker]).float().to(self.device)
                    else: _, emb_org = self.C(voice.squeeze(1).to(self.device),is_eval=True)#Was False
                    x_real = self.C(voice.squeeze(1).to(self.device),is_eval=True, only_preprocessing=True).transpose(1,2)#Was False
                    #print(emb_org.shape)
                    x_real = self.random_crop(x_real)
                    try:
                        val_loss = self.do_iteration(x_real, emb_org, train=False, cd=epoch>=self.use_loss_cd, emb=epoch>=self.use_loss_emb,  cs=epoch>=self.use_loss_cs)
                    except:
                        logging.error("Error happened during Validation iteration number "+str(epoch))
                        exit()
                    break
                
                # Print out validation information.
                log = "\tValidation [{}/{}]\t".format(epoch+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, val_loss[tag])
                logging.info(log)   
            
            if(False): #Test loss
                try:
                    voice1, _, _, speaker1, _ = next(test_loader)
                    voice2, _, _, speaker2, _ = next(test_loader)
                except:
                    test_loader = iter(self.test_loader)
                    voice1, _, _, speaker1, _ = next(test_loader)
                    voice2, _, _, speaker2, _ = next(test_loader)
                #print(voice1.shape)
                shape = min(voice1.shape[1], voice2.shape[1])
                voice, speaker = torch.cat((voice1[:,:shape],voice2[:,:shape]),dim=0), torch.cat((speaker1, speaker2), dim=0)
                #print(voice.shape)
                _, emb_org = self.C(voice.squeeze(1).to(self.device),is_eval=True)#Was False
                x_real = self.C(voice.squeeze(1).to(self.device),is_eval=True, only_preprocessing=True).transpose(1,2)#Was False
                x_real = self.random_crop(x_real)
                #print(x_real.shape, emb_org.shape)
                #try:
                val_loss = self.do_iteration(x_real, emb_org, train=False, cd=epoch>=self.use_loss_cd, emb=epoch>=self.use_loss_emb,  cs=epoch>=self.use_loss_cs)
                """except:
                    logging.error("Error happened during Evaluation iteration number "+str(epoch))
                    #exit()
                
                # Print out validation information.
                log = "\tEvaluation [{}/{}]\t".format(epoch+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, val_loss[tag])
                logging.info(log)   

            """
            if(epoch%10==0):#printing spectrums
                self.print_spectrum("train", epoch)
                self.print_spectrum("valid",   epoch)
                #self.print_spectrum("eval",  epoch)"""

            # Saving network
            if(self.save_model or (epoch+1)%100==0):
                self.save(self.savedir[:-5]+"_"+str(epoch+1)+self.savedir[-5:])
            if(eer_tgt <= best_eer_tgt):
                best_eer_tgt = eer_tgt
                self.save(self.savedir[:-5]+"_best"+self.savedir[-5:])
            self.save(self.savedir)
            logging.info("Saved at "+self.savedir)


        #self.test_zeros()
        self.save(self.savedir)
        logging.info("Last epoch saved at "+self.savedir)
        #self.test_zeros()
        #self.scorers.EER_post_generator_evaluation("train", self.C, self.G)
        self.scorers.EER_post_generator_evaluation("valid", self.C, self.G)
        self.scorers.EER_post_generator_evaluation("eval", self.C, self.G)
        """self.print_spectrum("train", epoch)
        self.print_spectrum("valid",   epoch)
        #self.print_spectrum("eval",  epoch)"""
        
    def print_spectrum(self,data_="train", epoch=0, batch_size=3):
        dataset = iter(self.scorers.get_dataset(data_))
        
        for idx, data in enumerate(dataset):
            if(idx==batch_size):break
            elif(idx==0): voices = data[0].unsqueeze(0)
            else: voices = torch.cat((voices, data[0].unsqueeze(0) ), dim=0)
            #print(voices.shape)
        spectrums = self.C(voices.squeeze(1).to(self.device), is_eval=True, only_preprocessing=True).transpose(1,2)
        _, embeddings = self.C(voices.squeeze(1).to(self.device), is_eval=True)
        post_spectrums, _, _ = self.G(spectrums,embeddings, embeddings)
        spectrums, post_spectrums = spectrums.cpu().detach().numpy(), post_spectrums.squeeze(1).cpu().detach().numpy()

        fig, axes = plt.subplots(nrows=spectrums.shape[0], ncols=2)
        for row in range(spectrums.shape[0]):
            axe_1, axe_2 = axes[row]

            axe_1.imshow(np.flip(spectrums[row].T, axis=1))
            if(row==0):axe_1.set_title("Spectrum before reconstruction")
            axe_2.imshow(np.flip(post_spectrums[row].T, axis=1))
            if(row==0):axe_2.set_title("Spectrum after reconstruction")
        #fig.set_title("Spectrums "+data+" - epoch "+str(epoch))
        
        filename = self.spectrums_dir+"_"+data_+"_"+str(epoch)
        fig.savefig(filename+".png")


    def test_zeros(self):
        #testing with zero matrices
        for data in self.train_loader:
            voice, _ = data
            voice = torch.zeros_like(voice.squeeze(1)).to(self.device)
            _, emb_org = self.C(voice,is_eval=True)
            emb_org = torch.zeros_like(emb_org)
            x_real = self.C(voice,is_eval=True, only_preprocessing=True).transpose(1,2)
            x_real = self.random_crop(x_real)
            x_identic, _, _ = self.G(torch.zeros_like(x_real),emb_org ,emb_org )
            #print(emb_org[0,:9], x_real.shape)
            #print(x_identic.shape, emb_org.shape, x_real.shape)
            #print(x_identic[0,0,0,:9])
            break

                
    def save(self,filename=""):
        if(len(filename)==0):
            filename=self.savedir
            logging.info("no saving file given, using : "+str(filename))
        torch.save({"model": self.G.state_dict()}, filename)
    
    def load(self, filename=""):
        if(len(filename)==0):
            filename=self.savedir
            logging.info("no loading file given, using : "+str(filename))
        model = torch.load(filename, map_location=self.device)
        self.G.load_state_dict(model["model"])
        self.G.eval().to(self.device)

    def random_crop(self, x_real):
        if(x_real.size(1)>self.len_crop):
            starts = np.random.randint(0,x_real.size(1)-self.len_crop-1, size=x_real.size(0))
            all_indx = starts[:,None] + np.arange(self.len_crop)
            x_real = x_real[np.arange(all_indx.shape[0])[:,None], all_indx,:]
        return x_real
    

        
        
