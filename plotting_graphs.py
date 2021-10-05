import numpy as np 
import matplotlib.pyplot as plt
import argparse

def extract_from_file(filename, no_eer_tr=True):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    #print(len(lines))
    EERs = {"tgt_train":[], "tgt_valid":[], "tgt_eval":[], "src_train":[], "src_valid":[], "src_eval":[]}
    losses = []
    use_eval_loss = False
    for line in lines:
        if "Evaluation" in line: use_eval_loss = True

    print("evaluation used : ",use_eval_loss)
    for idx,line in enumerate(lines):
        try:
            if("EER" in line):
                if("target val" in line):     EERs["tgt_valid"].append(float(line.split(":")[1].split("%")[0]))
                elif("source val" in line):   EERs["src_valid"].append(float(line.split(":")[1].split("%")[0]))
                elif("target train" in line): EERs["tgt_train"].append(float(line.split(":")[1].split("%")[0]))
                elif("source train" in line): EERs["src_train"].append(float(line.split(":")[1].split("%")[0]))
                elif("target test" in line):  EERs["tgt_eval"].append(float(line.split(":")[1].split("%")[0]))
                elif("source test" in line):  EERs["src_eval"].append(float(line.split(":")[1].split("%")[0]))

            elif(line[:7]=="Elapsed"):
                loss = [float(i.split(":")[1]) for i in line.strip().split(']')[-1].split(',')[1:]]
            elif("Validation" in line):
                loss += [float(i.split(":")[1]) for i in line.strip().split(']')[-1].split(',')[1:]]
                if (not use_eval_loss):
                    loss += [0]*(len(loss)//2)
                    losses.append(np.array(loss))
            elif("Evaluation" in line):
                loss += [float(i.split(":")[1]) for i in line.strip().split(']')[-1].split(',')[1:]]
                losses.append(np.array(loss))
        except:
            print("Error with line", idx)
            print(line)
    losses_ = np.zeros((len(losses), np.max([len(i) for i in losses])))
    for i, loss in enumerate(losses):
        if(len(loss)==losses_.shape[1]):
            losses_[i,:len(loss)] = loss
        else:
            losses_[i,:len(loss)//2] = loss[:len(loss)//2]
            losses_[i,losses_.shape[1]//2:(losses_.shape[1]//2)+(len(loss)//2)] = loss[len(loss)//2:]
            
    EERs = {key:np.array(list_) for key,list_ in EERs.items() if len(list_)>1}
    #print(losses)
    print( losses_.shape, EERs.keys())
    return EERs, losses_


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='800ep_halfresnetAnthony', help='log file for the experiment')
    args = parser.parse_args()
    no_eer_tr = True
    EERs, losses = extract_from_file("logs/"+args.logfile+".log", no_eer_tr=no_eer_tr)
    """if(losses.shape[1]==8):
        losses[:,2]*=10
        losses[:,6]*=10
        losses[:,3]*=100
        losses[:,7]*=100
    else:
        losses[:,2]*=10
        losses[:,5]*=10"""

    emb_loss = 1 if losses.shape[1]>9 else 0

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2+emb_loss, 1, 2+emb_loss)
    X = np.arange(len(EERs["tgt_valid"]))
    for key, data in EERs.items():
        ax1.plot(X, data, label="EER "+str(key))
    ax1.set_title("EERs")
    ax1.legend(loc='center right')

    X = np.arange(len(losses))
    print(losses.shape)
    ax2 = fig.add_subplot(2+emb_loss, 1, 1)
    ax2.plot(X, losses[:,0], label="train_loss_id", color="C0", linestyle="-")
    ax2.plot(X, losses[:,1], label="train_loss_id_psnt", color="C1", linestyle="-")
    ax2.plot(X, losses[:,3+emb_loss], label="valid_loss_id", color="C2", linestyle="--")
    ax2.plot(X, losses[:,4+emb_loss], label="valid_loss_id_psnt", color="C3", linestyle="--")
    ax2.plot(X, losses[:,6+ 2*emb_loss], label="eval_loss_id", color="C4", linestyle="-")
    ax2.plot(X, losses[:,7+ 2*emb_loss], label="eval_loss_id_psnt", color="C5", linestyle="-")
    
        
    ax2.plot(X, np.zeros_like(X), "k-")
    ax2.set_title("Losses id")
    ax2.legend()
    if(emb_loss==1):
        ax3 = fig.add_subplot(2+emb_loss, 1, 1+emb_loss)
        ax3.plot(X, 10*losses[:,3], label="train_loss_emb", color="C0", linestyle="-")
        ax3.plot(X, 10*losses[:,6], label="valid_loss_emb", color="C2", linestyle="--")
        ax3.plot(X, 10*losses[:,9], label="eval_loss_emb", color="C4", linestyle="--")
        ax2.plot(X, np.zeros_like(X), "k-")
        ax3.set_title("Losses emb")
        ax3.legend()
    """ax3 = fig.add_subplot(312)
    ax3.plot(X, losses[:,2], label="train_loss_cd", color="C2", linestyle="-")
    ax3.plot(X, losses[:,losses.shape[1]//2 +2], label="valid_loss_cd", linestyle="--")
    ax3.set_title("Losses cd")
    ax3.legend()"""
    """
    ax4 = fig.add_subplot(311)
    ax4.plot(X, losses[:,0], label="train_loss_id", color="C0", linestyle="-")
    ax4.plot(X, losses[:,1], label="train_loss_id_psnt", color="C1", linestyle="-")
    ax4.plot(X, losses[:,2], label="10Xtrain_loss_cd", color="C2", linestyle="-")
    if(losses.shape[1]>6):
        #ax4.plot(X, losses[:,3], label="100Xtrain_loss_emb", color="C3", linestyle="-")
        ax4.plot(X, losses[:,4], label="valid_loss_id", color="C0", linestyle="--")
        ax4.plot(X, losses[:,5], label="valid_loss_id_psnt", color="C1", linestyle="--")
        ax4.plot(X, losses[:,6], label="10Xvalid_loss_cd", color="C2", linestyle="--")
        #ax4.plot(X, losses[:,7], label="100Xvalid_loss_emb", color="C3", linestyle="--")
    else:
        ax4.plot(X, losses[:,3], label="valid_loss_id", color="C0", linestyle="--")
        ax4.plot(X, losses[:,4], label="valid_loss_id_psnt", color="C1", linestyle="--")
        ax4.plot(X, losses[:,5], label="10Xvalid_loss_cd", color="C2", linestyle="--")
    
    ax4.plot(X, np.zeros_like(X), "k-")
    ax4.set_title("Losses")
    ax4.legend()
    """
    """if(losses.shape[1]>6):
        ax5 = fig.add_subplot(324)
        ax5.plot(X, losses[:,3], label="train_loss_emb", color="C3", linestyle="-")
        ax5.plot(X, losses[:,7], label="valid_loss_emb", linestyle="--")
        ax5.plot(X, np.zeros_like(X), "k-")
        ax5.set_title("Losses embeddings")
        ax5.legend()"""

    plt.savefig("graphs/"+args.logfile+".png")
    #plt.show()