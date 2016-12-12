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
    resp, _ = util.response(1, stim, 1, 1)
    return np.append(stim, resp.T, 0)


# calculate the numerical probability and density of a bunch of stimuli and responses
def numericProbDens(stimRespArr, resolution=None, conZero=1):
    if resolution is None:
        resolution = [12, 12, 12]

    binEdges = genBins(conZero, resolution)

    counts, _ = np.histogramdd(stimRespArr.T, binEdges)
    probs = counts / np.shape(stimRespArr)[1]

    volumes = np.einsum('i,j,k->ijk', np.diff(binEdges[0]), np.diff(binEdges[1]), np.diff(binEdges[2]))
    dens = probs / volumes

    return binEdges, probs, dens


# edges: flag on whether to include edge cases, that can make calculating probabilities complicated, but assure
# that all values are covered
def genBins(conZero=1, resolution=None, edges=True):
    if resolution is None:
        resolution = [12, 12, 12]
    # generate histogram bins
    orispace = np.linspace(0, 2 * math.pi, resolution[1])

    minEdge = np.log10(prb.inverseExponCDF(10**-8, conZero))
    maxEdge = np.log10(prb.inverseExponCDF(1 - 10**-8, conZero))
    conspace = np.logspace(minEdge - 1, maxEdge + 1, resolution[0])
    respspace = np.logspace(minEdge, maxEdge+2, resolution[2])  # see plots for part of justification of values
    if edges:
        respspace[0] = 0
        respspace[-1] = np.inf
        conspace[0] = 0
        conspace[-1] = np.inf
    binEdges = [conspace, orispace, respspace]
    return binEdges


def marginalResponseIndependent(totalsamples, mult=1, respResolution=12, conZero=1):
    bins = genBins(conZero, [2, 2, respResolution])
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
    plt.xlabel('response rate')
    plt.ylabel('probability')
    acc /= np.diff(respspace)
    # print("diff rspace", np.diff(respspace))
    # print("acc", acc)
    # print("density integral:", np.einsum('i,i->i', acc[:-1], np.diff(respspace[:-1])))
    plt.figure()
    plt.plot(respspace[:-1], acc)
    plt.xscale('log')
    plt.xlabel('response rate')
    plt.ylabel('density')
    plt.figure()
    plt.plot(respspace[:-1], np.log(acc))
    plt.xscale('log')
    plt.xlabel('response rate')
    plt.ylabel('log density')
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


def analyitcPosteriorConOri(res1=36, res2=36, res3=36, conZero=1):
    bins = genBins(conZero, [res1, res3, res2], False)
    res = np.zeros([res1, res3, res2])
    for i, c in enumerate(bins[0]):
        for j, o in enumerate(bins[1]):
            for k, x in enumerate(bins[2]):
                res[i, j, k] = prb.pJointAll(x, o, c, conZero)
    volumes = np.einsum('i,j,k->ijk', np.diff(bins[0]), np.diff(bins[1]), np.diff(bins[2]))
    # this makes the densities into approximate probabilities (multiply density with step size)
    # approximate, because we're doing oversum, i.e. the probabilities are overestimated
    probs = np.einsum('ijk,ijk->ijk', res[:-1, :-1, :-1], volumes)
    # and marginalize over x (responses)
    margX = np.einsum('ijk->k', probs)
    # and divide by x probability
    post = probs[:, :, :] / margX[:]
    # print("num zeros in posterior:", np.sum(post == 0))
    # included this line for task one
    # maxes(bins, post, res)
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
        bins, probs, dens = numericProbDens(samples, [36, 36, 36])
    posts = numericPosteriorConOri(probs)
    print(np.size(posts), np.sum(posts == 0), np.sum(posts == np.inf))
    if plot:
        plotStuff(bins, posts)
    return posts


def analyticMarginalResponseProb(plotAll=False, conZero=1, plotMarg=False):
    d1 = 100
    d2 = 100
    bins = genBins(conZero, [d1, d1, d2])
    res = np.zeros([d1-1, d1-1, d2-1])
    for i, x in enumerate(bins[2][:-1]):
        for j, o in enumerate(bins[1][:-1]):
            for k, c in enumerate(bins[0][:-1]):
                res[k, j, i] = prb.pJointAll(x, o, c, 1)
    volumes = np.einsum('i,j,k->ijk', np.diff(bins[0][:-1]), np.diff(bins[1][:-1]), np.diff(bins[2][:-1]))

    # margX = np.einsum('ijk,ijk->k', res[1:, 1:, 1:], volumes)
    margX = np.einsum('ijk,ijk->k', res[:-1, :-1, :-1], volumes)      # probably the best one
    print(np.shape(margX), np.shape(volumes), np.shape(res), np.shape(bins[2]))
    if plotMarg:
        plt.semilogx(bins[2][:-2], margX)
        plt.xlabel('response rate')
        plt.ylabel('probability')
        margX /= np.diff(bins[2][:-1])
        plt.figure()
        plt.plot(bins[2][:-2], margX)
        plt.xscale('log')
        plt.xlabel('response rate')
        plt.ylabel('density')
        plt.figure()
        plt.plot(bins[2][:-2], np.log(margX))
        plt.xscale('log')
        plt.xlabel('response rate')
        plt.ylabel('log density')
        print("density integral?", np.einsum('i,i', margX, np.diff(bins[2][:-1])))
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
        ax.set_xticks(np.log10(bins[0][1:-1:5]))
        ax.set_xticklabels(map("{:.1e}".format, bins[0][1:-1:5]))
        ax.view_init(30, 145)
        ax.set_zlabel("posterior probability")
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


def taskSix(cZeros, tauZeros, windowSizes, numChans=8, numStim=100000, postRes=None):
    if postRes is None:
        postRes = [36, 36, 36]
    X, Y = np.meshgrid(cZeros, tauZeros)
    bestWS = X.T * 0
    for i, c in enumerate(cZeros):
        print("cZero:", c)
        bins, posts = analyitcPosteriorConOri(postRes[0], postRes[1], postRes[2], c)
        posts = logolize(posts)
        for j, t in enumerate(tauZeros):
            bestWS[i, j] = optWindowSize(c, t, windowSizes, numChans, numStim, bins, posts)
    return X, Y, bestWS


def logolize(arr):
    rtn = np.log(arr)
    selector = rtn == -np.inf
    rtn[selector] = np.amin(rtn[~selector])
    return rtn


def optWindowSize(cZero, tauZero, candidates, numChans, numStim, bins, posts):
    stm, durs = util.genStimuli(cZero, tauZero, numStim, 1)
    resp, prefer = util.response(numChans, stm, 1, 1)
    pstW1 = infr.accumTime(bins, posts, resp, prefer, 1)
    pstW1 = np.cumsum(pstW1, 0)
    pst = np.copy(pstW1)
    mn = np.inf
    bestWS = np.inf
    print("#######################################################ü")
    for c in candidates[::-1]:
        pst[c:] = pstW1[c:] - pstW1[:-c]
        inferred = infr.inferNaive(bins, pst)
        errors = infr.accuracy(stm, inferred, cZero)
        totalError = np.sum(errors)
        print("min so far, next error", mn, totalError)
        if totalError < mn:
            mn = totalError
            bestWS = c
    return bestWS


# function to test if the accumulation function seems to be working
def testAccum():
    stms = np.array([[1, 1, 0.1, 0.001, 1, 1, 1, .001], [0, 0, 0, np.pi, np.pi, np.pi, 1, 1]])
    bn, pst = analyitcPosteriorConOri()
    pst = logolize(pst)
    rsp, pref = util.response(36, stms, 1, 1)
    print(np.shape(rsp), np.shape(pref))
    print(len(rsp), len(pref))
    # print(list(enumerate(rsp)))
    accChan = infr.accumChannels(bn, pst, rsp[0], pref)
    print(infr.inferNaive(bn, accChan[np.newaxis, :, :]))
    quick3DPlots(accChan)


def quick3DPlots(arr):
    shp = np.shape(arr)
    X, Y = np.meshgrid(np.arange(shp[0]), np.arange(shp[1]))
    if len(shp) == 2:
        quick3DPlotHelp(X, Y, arr)
    else:
        for i in range(shp[2]):
            quick3DPlotHelp(X, Y, arr[:, :, i])
    plt.show()


def quick3DPlotHelp(X, Y, arr):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, arr, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)


def genDataTaskSix():
    # cZeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tauZeros = [1, 5, 10, 30, 50, 100, 300, 500, 800, 1000]
    cZeros = tauZeros[:]
    windowSizes = np.array([1, 5, 10, 50, 100, 500, 1000, 5000])
    x, y, z = taskSix(cZeros, tauZeros, windowSizes, 8, 10000)
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y = np.meshgrid(np.arange(len(cZeros)), np.arange(len(tauZeros)))
    surf = ax.plot_surface(x, y, np.log(z.T), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    # surf = ax.plot_surface(X, Y, values[:, :-1, i], rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.xlabel("contrast")
    plt.ylabel("duration")
    ax.set_zlabel("optimal window size")
    ax.set_zticks(np.log(windowSizes))
    ax.set_zticklabels(windowSizes)
    ax.set_xticklabels(cZeros)
    ax.set_yticklabels(tauZeros)
    plt.show()


# analyitcPosteriorConOri()
# testAccum()
# analyticMarginalResponseProb(False)
def plotAnalyticPosterior():
    bn, pst = analyitcPosteriorConOri()
    exlow = pst[:, :, 15]
    exhigh = pst[:, :, 30]

    def pltPost(bin, arr):
        X, Y = np.meshgrid(np.log10(bin[0][:-1]), bin[1][:-1])
        print("shapes, post, x, y", np.shape(pst), np.shape(X), np.shape(Y))
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, arr.T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        ax.set_xlabel("contrast")
        ax.set_ylabel("orientation")
        ax.set_zlabel("posterior probability")
        ax.set_xticks(np.log10(bin[0][:-1:5]))
        ax.set_xticklabels(map("{:.1e}".format, bin[0][:-1:5]))
        ax.view_init(30, 145)

    pltPost(bn, exlow)
    pltPost(bn, exhigh)
    print("responses:", bn[2][15], bn[2][30])
    plt.show()

# taskThree(False, True)
# plotAnalyticPosterior()

genDataTaskSix()


def taskFour(quick=True):
    meanContrast = 1
    meanDuration = 8
    windowSize = 1
    thresh = 0.01
    if quick:
        bn, pst = analyitcPosteriorConOri(10, 10, 10, meanContrast)
        stm, durs = util.genStimuli(meanContrast, meanDuration, 1000, 1)
        rsp, prf = util.response(4, stm, 1, 1)
    else:
        bn, pst = analyitcPosteriorConOri(60, 60, 60, meanContrast)
        stm, durs = util.genStimuli(meanContrast, meanDuration, 100000, 1)
        rsp, prf = util.response(4, stm, 1, 1)      # otherwise used 12

    pst = logolize(pst)
    pstpst = infr.accumTime(bn, pst, rsp, prf, windowSize)
    inferred = infr.inferNaive(bn, pstpst)

    pLtCon, pLtDur, cs, ds = infr.probLatency(stm, inferred, durs, conZero=meanContrast, durRes=48, thresh=thresh)
    print(ds)
    ident = "_4chans_" + str(meanContrast) + "_" + str(meanDuration) + "_" + str(windowSize) + "_" + str(thresh)
    plt.figure("prob of con" + ident)
    plt.semilogx(cs[:-1], pLtCon[0])
    plt.xlabel("stimulus contrast")
    plt.ylabel("mean probability of detection")

    plt.figure("latency of contr" + ident)
    plt.semilogx(cs[:-1], pLtCon[1])
    plt.xlabel("stimulus contrast")
    plt.ylabel("mean latency of detection")

    plt.figure("prob of dur" + ident)
    plt.plot(ds[:-1], pLtDur[0, :-1])
    plt.xlabel("stimulus duration")
    plt.ylabel("mean probability of detection")

    plt.figure("latency of dur" + ident)
    plt.plot(ds[:-1], pLtDur[1, :-1], ds[:-1], ds[:-1]-1, 'r')
    plt.legend(["result", "worst"], loc='best')
    plt.xlabel("stimulus duration")
    plt.ylabel("mean latency of detection")
    plt.show()


def taskFive(quick=True):
    meanContrast = 1
    meanDuration = 8
    windowSize = 4
    numChans = 8
    if quick:
        bn, pst = analyitcPosteriorConOri(10, 10, 10, meanContrast)
        stm, durs = util.genStimuli(meanContrast, meanDuration, 1000, 1)
        rsp, prf = util.response(4, stm, 1, 1)
    else:
        bn, pst = analyitcPosteriorConOri(30, 30, 30, meanContrast)
        stm, durs = util.genStimuli(meanContrast, meanDuration, 250000, 1)
        rsp, prf = util.response(numChans, stm, 1, 1)      # using 12 or 4

    pst = logolize(pst)
    pstpst = infr.accumTime(bn, pst, rsp, prf, windowSize)
    inferred = infr.inferNaive(bn, pstpst)
    ident = "_highnum_" + str(meanContrast) + "_" + str(meanDuration) + "_" + str(windowSize) + "_" + str(numChans)

    # standard deviation performance
    # infr.performanceDiff(stm, inf)
    stdCon, stdDur, cs = infr.performanceSTD(stm, inferred, durs)
    # print(stdCon, stdDur, cs)
    # print(np.shape(cs), np.shape(stdCon), np.shape(stdDur))
    plt.figure("concon" + ident)
    plt.loglog(cs[:-1], stdCon[0])
    plt.xlabel("object contrast")
    plt.ylabel("standard deviation of contrast error")
    plt.figure("oricon"+ident)
    plt.semilogx(cs[:-1], stdCon[1])
    plt.xlabel("object contrast")
    plt.ylabel("standard deviation of orientation error")

    plt.figure("condur" + ident)
    durRange = np.arange(np.amax(durs))
    plt.plot(durRange, stdDur[0])
    plt.xlabel("object duration")
    plt.ylabel("standard deviation of contrast error")

    plt.figure("oridur" + ident)
    plt.plot(durRange, stdDur[1])
    plt.xlabel("object duration")
    plt.ylabel("standard deviation of orientation error")

    plt.show()

