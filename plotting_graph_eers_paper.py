import numpy as np 
import matplotlib.pyplot as plt

EER_src = np.array([48.15, 47.53, 48.14, 46.94, 41.04, 49.12, 43.20, 5.55, 1.23])
EER_tgt = np.array([18.51, 12.34,  9.94,  5.22,  3.37,  0.25, 19.44, 46.06, 47.25])
X = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128])
Y = np.arange(len(X))
#EER_src, EER_tgt, X = EER_src[1:], EER_tgt[1:], X[1:]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Y, EER_src, ".-", color="blue", label="EER source")
ax.plot(Y, EER_tgt, "+--", color="red", label="EER target")
ax.legend()
ax.set_xlabel("Bottleneck values")
ax.set_ylabel("EER")
ax.set_title("EER source and target for different bottlenecks values")
#ax.set_xscale("log")
plt.xticks(ticks=Y,labels=X)
ax.grid(True)
plt.tight_layout(pad=1)
plt.savefig("eer_graph.eps")