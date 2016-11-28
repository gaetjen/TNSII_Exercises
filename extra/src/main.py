import math
import numpy as np
import scipy.io
import scipy.stats as stat
import helpers as util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# stim = util.genStimuli(1, 10, 100000, 1)
# resp = util.response(1, stim, 1, 1)

joint100 = np.array(scipy.io.loadmat('joint100.mat')['joint'])

print(np.shape(joint100))

ns = 100

if True:
    conSamples = np.sort(np.random.exponential(1, ns))
    oriSamples = np.sort(np.random.random(ns) * 2 * math.pi)
    stim = np.zeros([2, ns])
    resp = np.zeros([ns, ns, ns])

    for i in range(ns):
        stim[1, :] = conSamples[i]
        for j in range(ns):
            stim[0, :] = oriSamples[j]
            resp[i, j, :] = util.response(1, stim, 1, 1)[:, 0]
        if i % 10 == 0:
            print(i)

densities = stat.expon.pdf(conSamples) / (2 * math.pi)
# repmat?

conspace = np.logspace(math.log(min(conSamples), 10), math.log(max(conSamples), 10), 36)
orispace = np.linspace(0, 2*math.pi, 36)
respspace = np.logspace(math.log(np.amin(resp), 10), math.log(np.amax(resp), 10), 24)

if False:
    joint = np.zeros([ns, ns, len(respspace)-1])

    for i in range(ns):
        for j in range(ns):
            respDensity = np.histogram(resp[i, j, :], respspace)
            # respDensity /= np.trapz(respDensity)      # don't make into density for now, too much headache
            joint[i, j, :] = respDensity[0]
        # joint[i, :, :] *= densities[i]
        if i % 40 == 0:
            print(i)


    discreteProb = np.zeros([len(conspace) - 1, len(orispace) - 1, len(respspace) - 1])
    for i in range(1, len(conspace)-1):
        print(i)
        for j in range(1, len(orispace)-1):
            conSelector = (conSamples < conspace[i]) & (conSamples >= conspace[i-1])
            oriSelector = (oriSamples < orispace[j]) & (oriSamples >= orispace[j-1])
            subspace = joint[np.ix_(conSelector, oriSelector)]                               # fails when there are no values, try checking first
            sum1 = np.sum(subspace, 0)
            discreteProb[i, j, :] = sum(sum1, 1)


if True:
    X, Y = np.meshgrid(np.log10(conspace[1:]), orispace[1:])

    # for i in range(np.shape(discreteProb)[-1]):
    for i in range(0, len(respspace), 3):
        fig = plt.figure(str(respspace[i]))
        ax = Axes3D(fig)
        # print(np.shape(X), np.shape(Y), np.shape(discreteProb[:, :, i]))
        surf = ax.plot_surface(X, Y, np.transpose(discreteProb[:, :, i]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        plt.xlabel("contrast")
        plt.ylabel("orientation")
        ax.set_xticks(np.log10(conspace[1:]))
        ax.set_xticklabels(map("{:.1e}".format, conspace[1:]))
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax.xaxis.get_ticklabels()[::4]:
            label.set_visible(True)
    plt.show()

if False:
    X, Y = np.meshgrid(np.log10(conspace[1:]), np.log10(respspace[1:]))

    # for i in range(np.shape(discreteProb)[-1]):
    for i in range(0, 35,3):
        fig = plt.figure(i)
        ax = Axes3D(fig)
        # print(np.shape(X), np.shape(Y), np.shape(discreteProb[:, :, i]))
        vals = discreteProb[:, i, :]
        surf = ax.plot_surface(X, Y, np.transpose(vals), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        plt.xlabel("contrast")
        plt.ylabel("response")
        ax.set_xticks(np.log10(conspace[1:]))
        ax.set_xticklabels(map("{:.1e}".format, conspace[1:]))
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax.xaxis.get_ticklabels()[::4]:
            label.set_visible(True)
        ax.set_yticks(np.log10(respspace[1:]))
        ax.set_yticklabels(respspace[1:])

    plt.show()

idx = np.argmax(joint)
maxConI, maxOriI, maxXI = np.unravel_index(idx, np.shape(joint))
maxes = [conSamples[maxConI], oriSamples[maxOriI], respspace[maxXI]]
print(maxConI, maxOriI, maxXI, "\n", maxes)

idx = np.argmax(discreteProb)
maxConI, maxOriI, maxXI = np.unravel_index(idx, np.shape(discreteProb))
maxes = [conspace[maxConI], orispace[maxOriI], respspace[maxXI]]
print(maxConI, maxOriI, maxXI, "\n", maxes)





print("done")
