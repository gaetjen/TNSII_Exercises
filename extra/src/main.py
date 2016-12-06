import math
import numpy as np
import scipy.io
import scipy.stats as stat
import helpers as util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize as opt
import probabilities as prb
import inference as infr


# generates a bunch of random stimuli and their responses
def genResponses(numstimuli, conZero=1):
    conSamples = np.random.exponential(conZero, numstimuli)
    oriSamples = np.random.random(numstimuli) * 2 * np.pi
    stim = np.array([conSamples, oriSamples])
    resp = util.response(1, stim, 1, 1)
    return np.append(stim, resp.T, 0)


# calculate the numerical probability and density of a bunch of stimuli and responses
def numericProbDens(stimRespArr, resolution=None):
    if resolution is None:
        resolution = [12, 12, 12]

    binEdges = genBins(resolution)

    counts, _ = np.histogramdd(stimRespArr.T, binEdges)
    probs = counts / np.shape(stimRespArr)[1]

    volumes = np.einsum('i,j,k->ijk', np.diff(binEdges[0]), np.diff(binEdges[1]), np.diff(binEdges[2]))
    dens = probs / volumes

    return binEdges, probs, dens


# edges: flag on whether to include edge cases, that can make calculating probabilities complicated, but assure
# that all values are covered
def genBins(resolution=None, edges=True):
    if resolution is None:
        resolution = [12, 12, 12]
    # generate histogram bins
    respspace = np.logspace(-8, 3, resolution[2])  # see plots for part of justification of values
    orispace = np.linspace(0, 2 * math.pi, resolution[1])
    conspace = np.logspace(np.log10(2 * 10 ** -9), np.log10(200), resolution[0])
    if edges:
        respspace[0] = 0
        respspace[-1] = np.inf
        conspace[0] = 0
        conspace[-1] = np.inf
    binEdges = [conspace, orispace, respspace]
    return binEdges


def marginalResponseIndependent(totalsamples, mult=1, respResolution=12):
    bins = genBins([2, 2, respResolution])
    respspace = bins[2]
    acc = respspace[:-1] * 0
    for i in range(mult):
        if i % 50 == 0:
            print(i)
        resp = genResponses(totalsamples)[2]
        margX, _ = np.histogram(resp, respspace)
        acc = acc + margX
    print("total theo, total actual: ", totalsamples * mult, np.sum(acc))
    acc /= (totalsamples * mult)
    print("after:", np.sum(acc))
    plt.plot(respspace[:-1], acc)
    plt.xscale('log')
    plt.xlabel('response rate (actual data slightly shifted right)')
    plt.ylabel('probability')
    acc /= np.diff(respspace)
    plt.figure()
    plt.plot(respspace[:-1], margX)
    plt.xscale('log')
    plt.xlabel('response rate (actual data slightly shifted right)')
    plt.ylabel('density')
    plt.show()


def maxes(bins, probs, dens):
    # get maximum probability
    idx = np.argmax(probs)
    print("maximum:\n\t", np.amax(probs))
    maxConI, maxOriI, maxXI = np.unravel_index(idx, np.shape(probs))
    m = [bins[0][maxConI], bins[1][maxOriI], bins[2][maxXI]]
    print("probabilities:\n\t", maxConI, maxOriI, maxXI, "\n\t", m)

    # get maximum density
    idx = np.argmax(dens)
    print("maximum:\n\t", np.amax(dens))
    maxConI, maxOriI, maxXI = np.unravel_index(idx, np.shape(dens))
    m = [bins[0][maxConI], bins[1][maxOriI], bins[2][maxXI]]
    print("densities:\n\t", maxConI, maxOriI, maxXI, "\n\t", m)


# posterior probability of contrast and orientation, given response
def numericPosteriorConOri(probs):
    marginalResp = np.einsum('ijk->k', probs)
    # print(np.shape(probs), np.shape(marginalResp))
    print(marginalResp)
    rtn = probs
    nonzeros = marginalResp != 0
    rtn[:, :, ~nonzeros] = 0
    rtn[:, :, nonzeros] = rtn[:, :, nonzeros] / marginalResp[nonzeros]
    return rtn


def analyitcPosteriorConOri(res1=36, res2=36, conZero=1):
    bins = genBins([res1, res1, res2], False)
    res = np.zeros([res1, res1, res2])
    for i, c in enumerate(bins[0]):
        for j, o in enumerate(bins[1]):
            for k, x in enumerate(bins[2]):
                res[i, j, k] = prb.pJointAll(x, o, c, conZero)
    volumes = np.einsum('i,j,k->ijk', np.diff(bins[0]), np.diff(bins[1]), np.diff(bins[2]))
    # this makes the densities into approximate probabilities (multiply density with step size)
    # approximate, because we're doing oversum, i.e. the probabilities are overestimated
    # NOTE: not 100% understood why we take contrast 1: instead of :-1, but it gives better results
    probs = np.einsum('ijk,ijk->ijk', res[1:, :-1, :-1], volumes)
    # and marginalize over x (responses)
    margX = np.einsum('ijk->k', probs)
    # and divide by x probability
    post = probs[:, :, 1:] / margX[:-1]
    # print("num zeros in posterior:", np.sum(post == 0))
    return bins, post


def taskOne(quick=True, plotP=False, plotD=False):
    if quick:
        samples = genResponses(100000)
        bins, probs, dens = numericProbDens(samples)
    else:
        samples = genResponses(10000000)
        bins, probs, dens = numericProbDens(samples, [24, 36, 24])
    if plotP:
        plotStuff(bins, probs)
    if plotD:
        plotStuff(bins, dens)
    maxes(bins, probs, dens)


def taskTwo(numsamples=100000, mult=1, res=12):
    marginalResponseIndependent(numsamples, mult, res)


def taskThree(quick=True, plot=False):
    if quick:
        samples = genResponses(100000)
        bins, probs, dens = numericProbDens(samples)
    else:
        samples = genResponses(100000000)
        bins, probs, dens = numericProbDens(samples, [12, 12, 12])
    posts = numericPosteriorConOri(probs)
    print(np.size(posts), np.sum(posts == 0), np.sum(posts == np.inf))
    if plot:
        plotStuff(bins, posts)
    return posts


def analyticMarginalResponseProb(plotAll=False):
    d1 = 100
    d2 = 100
    bins = genBins([d1, d1, d2])
    res = np.zeros([d1-1, d1-1, d2-1])
    for i, x in enumerate(bins[2][:-1]):
        for j, o in enumerate(bins[1][:-1]):
            for k, c in enumerate(bins[0][:-1]):
                res[k, j, i] = prb.pJointAll(x, o, c, 1)
    volumes = np.einsum('i,j,k->ijk', np.diff(bins[0][:-1]), np.diff(bins[1][:-1]), np.diff(bins[2][:-1]))

    # margX = np.einsum('ijk,ijk->k', res[1:, 1:, 1:], volumes)
    margX = np.einsum('ijk,ijk->k', res[:-1, :-1, :-1], volumes)      # probably the best one
    print(np.shape(margX), np.shape(volumes), np.shape(res), np.shape(bins[2]))
    plt.semilogx(bins[2][:-2], margX)

    print(np.einsum('i,i', margX, np.diff(bins[2][:-1])))
    plt.show()
    if plotAll:
        plotStuff(bins, res)


def plotStuff(bins, values):
    X, Y = np.meshgrid(np.log10(bins[0][1:-1]), bins[1][:-1])
    print(np.shape(values), np.shape(X))
    for i, v in enumerate(bins[2][:-1]):
        # print(posts[:, :, i])
        print(i, str(v))
        fig = plt.figure(str(v))
        ax = Axes3D(fig)
        surf = ax.plot_surface(X, Y, np.transpose(values[:-1, :, i]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        plt.xlabel("contrast")
        plt.ylabel("orientation")
        ax.set_xticks(np.log10(bins[0][1:-1]))
        ax.set_xticklabels(map("{:.1e}".format, bins[0][1:-1]))
        plt.show()
        #         for label in ax.xaxis.get_ticklabels():
        #             label.set_visible(False)
        #         for label in ax.xaxis.get_ticklabels()[::4]:
        #             label.set_visible(True)


def plotPost(bins, values):
    X, Y = np.meshgrid(np.log10(bins[0][1:-1]), bins[1][1:-1])
    print(np.shape(values), np.shape(X))
    for i, v in enumerate(bins[2][1:-1]):
        # print(posts[:, :, i])
        print(i, str(v))
        fig = plt.figure(str(v))
        ax = Axes3D(fig)
        surf = ax.plot_surface(X, Y, np.transpose(values[:, :, i-1]), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        # surf = ax.plot_surface(X, Y, values[:, :-1, i], rstride=1, cstride=1, cmap=cm.coolwarm)
        plt.xlabel("contrast")
        plt.ylabel("orientation")
        ax.set_xticks(np.log10(bins[0][1:-1]))
        ax.set_xticklabels(map("{:.1e}".format, bins[0][1:-1]))
        plt.show()

# b, pst = analyitcPosteriorConOri()
# print("shape of bins, post:", np.shape(b), np.shape(pst))
# plotPost(b, pst)
# taskTwo(1000000, 100, 100)
# analyticMarginalResponseProb()
# analyticStuff()
bn, pst = analyitcPosteriorConOri(36, 36, 1)
# plotStuff(bn, pst)
# print(bn[1])
print("num zero in pst", np.sum(pst == 0))
print("smallest real prob", np.amin(pst[pst != 0]))
pst = np.log(pst)
print("smallest log prob, max", np.amin(pst[pst != -np.inf]), np.amax(pst))
# zero is not a probability! great example of how, when your prior is P(x) = 0, no amount of evidence can
# convince you of it being true --> make prior equal to smallest probability otherwise
selector = pst == -np.inf
pst[selector] = np.amin(pst[~selector])
print("min, max", np.amin(pst), np.amax(pst))

# input("cont")
tl = 10000
stm, durs = util.genStimuli(1, 20, tl, 1)
print("generated stims")
rsp, prf = util.response(4, stm, 1, 1)
print("generated responses")
# speculation: accumTime function not good enough, or posterior prob
pstpst = infr.accumTime(bn, pst, rsp, prf, 5)
print("accumulated information over time")
print("infs, nans, tot:", np.sum(pstpst == -np.inf), np.sum(np.isnan(pstpst)), np.size(pstpst))
inf = infr.inferNaive(bn, pstpst)
print("inferred stimuli")
print("total difference", np.sum(np.abs(stm - inf)))
# print(np.abs(stm - foo))
t = np.arange(tl)
# plt.plot(t, stm[0], t, inf[0], )
# plt.legend(['stim', 'inferred'])
# plt.figure()
# plt.plot(t, np.sign(stm[0] - inf[0]))
# plt.figure()
# plt.plot(t, stm[1], t, inf[1])
# plt.legend(['stim', 'inferred'])
# print(infr.performanceIdcs(stm, inf, bn))

# infr.performanceDiff(stm, inf)
stdCon, stdDur, cs = infr.performanceSTD(stm, inf, durs)
# print(stdCon, stdDur, cs)
# print(np.shape(cs), np.shape(stdCon), np.shape(stdDur))
plt.loglog(cs[:-1], stdCon[0])
plt.legend(['contrast'])
plt.figure()
plt.semilogx(cs[:-1], stdCon[1])
plt.figure()
durRange = np.arange(np.amax(durs))
plt.plot(durRange, stdDur[0])
plt.figure()
plt.plot(durRange, stdDur[1])
plt.legend(['orientation'])
plt.show()
print("done")
