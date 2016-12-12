

def genResponses(numsamples):
    # generate lots of stimuli
    conSamples = np.sort(np.random.exponential(1, numsamples))
    oriSamples = np.sort(np.random.random(numsamples) * 2 * math.pi)
    stim = np.zeros([2, numsamples])

    # generate lots of responses for all the stimuli
    resp = np.zeros([numsamples, numsamples, numsamples])
    for i in range(numsamples):
        stim[1, :] = conSamples[i]
        for j in range(numsamples):
            stim[0, :] = oriSamples[j]
            resp[i, j, :] = util.response(1, stim, 1, 1)[:, 0]  # one channel, orientation = 0 degrees
        if i % 40 == 0:
            print(i)
    return resp, conSamples, oriSamples


# samples are not independent!
# compute solution for task two
def marginalResponse(numsamples, respResolution=12):
    resp, _, _ = genResponses(numsamples)
    acc, respspace = accumulateResponses(resp, respResolution)
    # print(np.shape(acc))
    acc = np.einsum('ijk->k', acc)
    acc /= np.sum(acc)
    # print(np.shape(acc))
    # print(np.shape(respspace))
    acc /= np.diff(respspace)
    print(np.einsum('i,i', acc, np.diff(respspace)))
    plt.plot(respspace[:-1], acc)
    plt.xscale('log')
    plt.xlabel('response rate (actual data slightly shifted right)')
    plt.ylabel('density')
    plt.show()


def accumulateResponses(resp, respResolution):
    # discretize into intervals

    respspace = np.logspace(np.log10(np.amin(resp)), np.log10(np.amax(resp)), respResolution)
    numsamples = np.shape(resp)[0]
    # histogramize responses for every orientation/contrast combination
    counts = np.zeros([numsamples, numsamples, len(respspace) - 1])
    for i in range(numsamples):
        for j in range(numsamples):
            respProb, _ = np.histogram(resp[i, j, :], respspace)
            counts[i, j, :] = respProb
        if i % 40 == 0:
            print(i)
    return counts, respspace


# compute solution for task one
# spaceResolution: [contrast, orientation, response]
def maxDens(numsamples, spaceResolution=[24, 18, 12]):
    # do stuff to get a joint distribution (first probability, in intervals, then density from there?)
    resp, conSamples, oriSamples = genResponses(numsamples)

    counts, respspace = accumulateResponses(resp, spaceResolution[2])

    # logspace takes the exponents as arguments
    conspace = np.logspace(np.log10(min(conSamples)), np.log10(max(conSamples)), spaceResolution[0])
    orispace = np.linspace(0, 2 * math.pi, spaceResolution[1])

    # histogramize over contrast and orientation
    probs = np.zeros([len(conspace) - 1, len(orispace) - 1, len(respspace) - 1])
    for i in range(1, len(conspace) - 1):
        # print(i)
        for j in range(1, len(orispace) - 1):
            conSelector = (conSamples < conspace[i]) & (conSamples >= conspace[i - 1])
            oriSelector = (oriSamples < orispace[j]) & (oriSamples >= orispace[j - 1])
            subspace = counts[np.ix_(conSelector, oriSelector)]
            sum1 = np.sum(subspace, 0)
            probs[i, j, :] = sum(sum1, 1)

    # divide by total sum for probability
    print(np.sum(probs))
    probs /= np.sum(probs)

    # divide by cell volume for density
    volumes = np.einsum('i,j,k->ijk', np.diff(conspace), np.diff(orispace), np.diff(respspace))
    densities = probs / volumes
    print(np.sum(probs))
    print(np.sum(volumes))
    print(np.sum(densities))
    # get maximum probability
    idx = np.argmax(probs)
    maxConI, maxOriI, maxXI = np.unravel_index(idx, np.shape(probs))
    maxes = [conSamples[maxConI], oriSamples[maxOriI], respspace[maxXI]]
    print("probabilities\n\t", maxConI, maxOriI, maxXI, "\n\t", maxes)

    # get maximum density
    idx = np.argmax(densities)
    print("maximum:\n\t", np.amax(densities))
    maxConI, maxOriI, maxXI = np.unravel_index(idx, np.shape(densities))
    maxes = [conspace[maxConI], orispace[maxOriI], respspace[maxXI]]
    print(conspace[maxConI + 1], orispace[maxOriI + 1], respspace[maxXI + 1])

    return maxes


def analyticPDF(resp, contrast, ori, conZero=1, tuningWidth=1, sensitivity=1, channelOri=0):
    meanResp = sensitivity * contrast * math.exp(2 * math.cos(tuningWidth * (channelOri-ori)))
    constants = np.log(math.pi/2) - np.log(conZero) - np.log(2*math.pi)
    # print(constants)
    simples = - math.pi * resp * resp / (4 * meanResp * meanResp) - contrast/conZero
    logs = np.log(resp) - 2 * np.log(meanResp)
    return math.exp(constants + simples + logs), simples, logs

# numsamples = 100
# mx = maxDens(100)
# print("numerical: ", mx)
# print("analytic: ", analyticPDF(mx[0], mx[1], mx[2]))
# # results do NOT fit with theory.
# # x should be ~0.4, is 10^-3
# # theta should be 0, is pi
# # c should be ~1, is 10^-2
# print("theoretic: ", analyticPDF(0.4, 1, 0))



if False:
    # b, pst = analyitcPosteriorConOri()
    # print("shape of bins, post:", np.shape(b), np.shape(pst))
    # plotPost(b, pst)
    # taskTwo(1000000, 100, 100)
    # analyticMarginalResponseProb()
    # analyticStuff()
    cc = 1
    bn, pst = analyitcPosteriorConOri(60, 60, cc)
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
    tl = 80000
    stm, durs = util.genStimuli(cc, 10, tl, 1)
    print("generated stims")
    rsp, prf = util.response(12, stm, 1, 1)
    print("generated responses")
    pstpst = infr.accumTime(bn, pst, rsp, prf, 3)
    print("accumulated information over time")
    print("infs, nans, tot:", np.sum(pstpst == -np.inf), np.sum(np.isnan(pstpst)), np.size(pstpst))
    inf = infr.inferNaive(bn, pstpst)
    print("inferred stimuli")
    print("total difference", np.sum(np.abs(stm - inf)))
    # infr.performanceIdcs(stm, inf, bn)
    # print(np.abs(stm - foo))
    t = np.arange(tl)
    if False:
        # plotting stimuli
        plt.plot(t, stm[0], t, inf[0], )
        plt.legend(['stim', 'inferred'])
        plt.figure()
        # plt.plot(t, np.sign(stm[0] - inf[0]))
        plt.figure()
        plt.plot(t, stm[1], t, inf[1])
        plt.legend(['stim', 'inferred'])
        # print(infr.performanceIdcs(stm, inf, bn))
    if False:

    if False:
        # mean transformed/normalized differences
        stdCon, stdDur, cs = infr.performanceDiff(stm, inf, durs, cc)
        # print(stdCon, stdDur, cs)
        # print(np.shape(cs), np.shape(stdCon), np.shape(stdDur))
        plt.loglog(cs[:-1], stdCon[0])
        plt.legend(['contrast'])
        plt.figure()
        plt.loglog(cs[:-1], stdCon[1])
        plt.figure()
        durRange = np.arange(np.amax(durs))
        plt.plot(durRange, stdDur[0])
        plt.figure()
        plt.plot(durRange, stdDur[1])
        plt.legend(['orientation'])
    if True:
        # latency and duration performance
        pLtCon, pLtDur, cs, ds = infr.probLatency(stm, inf, durs, conZero=cc)
        # print(stdCon, stdDur, cs)
        # print(np.shape(cs), np.shape(stdCon), np.shape(stdDur))
        plt.figure("prob of con")
        plt.semilogx(cs[:-1], pLtCon[0])
        plt.figure("latency of contr")
        plt.semilogx(cs[:-1], pLtCon[1])

        plt.figure("prob of dur")
        plt.plot(ds[:-1], pLtDur[0])
        plt.figure("latency of dur")
        plt.plot(ds[:-1], pLtDur[1], ds[:-1], ds[:-1] + 1, 'r')
        plt.legend(["result", "worst"])

    plt.show()
    print("done")