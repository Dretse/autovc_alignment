import numpy as np 
import matplotlib.pyplot as plt
import argparse

def extract_from_file(filename, no_eer_tr=True):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    #print(len(lines))
    EERs = {"tgt_train":[], "tgt_valid":[], "tgt_eval":[], "src_train":[], "src_valid":[], "src_eval":[], "src_back_test":[], "tgt_back_test":[]}
    losses = []

    use_eval_loss = False
    for line in lines:
        if "Evaluation" in line: 
            use_eval_loss = True
            break

    #print("evaluation used : ",use_eval_loss)
    for idx,line in enumerate(lines):
        try:
            if("EER" in line):
                if("target val" in line):     EERs["tgt_valid"].append(float(line.split(":")[1].split("%")[0]))
                elif("source val" in line):   EERs["src_valid"].append(float(line.split(":")[1].split("%")[0]))
                elif("target train" in line): EERs["tgt_train"].append(float(line.split(":")[1].split("%")[0]))
                elif("source train" in line): EERs["src_train"].append(float(line.split(":")[1].split("%")[0]))
                elif("target test" in line):  EERs["tgt_eval"].append(float(line.split(":")[1].split("%")[0]))
                elif("source test" in line):  EERs["src_eval"].append(float(line.split(":")[1].split("%")[0]))
                elif("target back_test" in line):  EERs["tgt_back_test"].append(float(line.split(":")[1].split("%")[0]))
                elif("source back_test" in line):  EERs["src_back_test"].append(float(line.split(":")[1].split("%")[0]))

            elif("Elapsed" in line):
                loss = [float(i.split(":")[1]) for i in line.strip().split(']')[-1].split(',')[1:]]
            elif("Validation" in line):
                loss += [float(i.split(":")[1]) for i in line.strip().split(']')[-1].split(',')[1:]]
                if (not use_eval_loss):
                    loss += [0]*(len(loss)//2)
                    losses.append(np.array(loss))
            elif("Evaluation" in line):
                loss += [float(i.split(":")[1]) for i in line.strip().split(']')[-1].split(',')[1:]]
            
            elif("ime taken" in line):losses.append(np.array(loss))
        except:
            None
            #print("Error with line", idx)
            #print(line)
    losses_ = np.zeros((len(losses), np.max([len(i) for i in losses])))
    for i, loss in enumerate(losses):
        if(len(loss)==losses_.shape[1]):
            losses_[i,:len(loss)] = loss
        else:
            losses_[i,:len(loss)//2] = loss[:len(loss)//2]
            losses_[i,losses_.shape[1]//2:(losses_.shape[1]//2)+(len(loss)//2)] = loss[len(loss)//2:]
            
    EERs = {key:np.array(list_) for key,list_ in EERs.items() if len(list_)>1}
    #print(losses)
    #print( losses_.shape, EERs.keys())
    return EERs, np.concatenate((losses_[:,1:],losses_[:,0][:,np.newaxis]),axis=1)

def plot_graph(logfile, no_eer_tr=True):
    EERs, losses = extract_from_file("logs/"+logfile+".log", no_eer_tr=no_eer_tr)
    emb_loss = 1 if losses.shape[1]>9 else 0

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2+emb_loss, 1, 2+emb_loss)
    #X = np.arange(len(EERs["tgt_valid"]))
    for key, data in EERs.items():
        try:
            ax1.plot(data, label="EER "+str(key))
        except:
            print(key)
    ax1.set_title("EERs")
    ax1.legend(loc='center right')

    X = np.arange(len(losses))
    print(losses.shape)
    ax2 = fig.add_subplot(2+emb_loss, 1, 1)
    ax2.plot(X, losses[:,0], label="train_loss_id", color="C0", linestyle="-")
    ax2.plot(X, losses[:,1], label="train_loss_id_psnt", color="C1", linestyle="-")
    #ax2.plot(X, losses[:,3+emb_loss], label="valid_loss_id", color="C2", linestyle="--")
    #ax2.plot(X, losses[:,4+emb_loss], label="valid_loss_id_psnt", color="C3", linestyle="--")
    #ax2.plot(X, losses[:,6+ 2*emb_loss], label="eval_loss_id", color="C4", linestyle="-")
    #ax2.plot(X, losses[:,7+ 2*emb_loss], label="eval_loss_id_psnt", color="C5", linestyle="-")
    
        
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

    plt.savefig("graphs/"+logfile+".png")
    #plt.show()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='500ep_model0', help='log file for the experiment')
    args = parser.parse_args()
    plot_graph(args.logfile)
    