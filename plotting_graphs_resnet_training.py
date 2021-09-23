import numpy as np 
import matplotlib.pyplot as plt
import argparse

def extract_from_file(filename, no_eer_tr=True):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    print(len(lines))
    Losses, Acc_train, Acc_valid, EER_valid, EER_test = [], [], [], [], []

    for idx,line in enumerate(lines):
        try:
            if "Epoch" in line:
                line_ = line.strip('\n').split('\t')[-2:]
                Losses.append(float(line_[0].split(':')[1]))
                Acc_train.append(float(line_[1].split(':')[1]))
            elif "Validation metrics" in line:
                line_ = line.strip("\n").split("-")[-1].split(",")
                Acc_valid.append(float(line_[0].split('=')[-1].strip('%')))
                EER_valid.append(float(line_[1].split('=')[-1].strip('%')))
            elif "Test metrics" in line:
                EER_test.append(float(line.strip("%\n").split("=")[-1]))
        except:
            print("Error with line", idx)
            print(line)
    
    Losses, Acc_train, Acc_valid, EER_valid, EER_test = np.array(Losses), np.array(Acc_train), np.array(Acc_valid), np.array(EER_valid), np.array(EER_test)
    print(Losses.shape, Acc_train.shape, Acc_valid.shape, EER_valid.shape, EER_test.shape)
    return [Losses, Acc_train, Acc_valid, EER_valid, EER_test]


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='training_resnet_0', help='log file for the experiment')
    args = parser.parse_args()
    no_eer_tr = True
    arguments = extract_from_file("logs/"+args.logfile+".log", no_eer_tr=no_eer_tr)
    arguments[1] = [i for idx,i in enumerate(arguments[1]) if idx%8==0]
    print(len(arguments[1]))
    fig = plt.figure(figsize=(10,10))
    X = np.arange(len(arguments[0]))
    Y = np.arange(len(arguments[-1]))
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(X, arguments[0], label="Loss")
    ax1.set_title("Loss")
    ax1.legend(loc='center right')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(Y, arguments[1], label="Acc_train")
    ax2.plot(Y, arguments[2], label="Acc_valid")
    ax2.plot(Y, arguments[3], label="EER_valid")
    ax2.plot(Y, arguments[4], label="EER_test")
    ax2.set_title("Acc + EER")
    ax2.legend(loc='center right')

    plt.savefig("graphs/"+args.logfile+".png")
    #plt.show()