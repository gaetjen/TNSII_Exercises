import numpy as np


# give the corresponding posterior distribution of stimulus parameters, given the response
def postsGivenResp(bins, posts, response, preferred=0):
    assert 0 <= preferred <= (2 * np.pi), "preferred should be in range 0, 2π"
    idx = np.argmin(np.abs(bins[2][:-1] - response))
    slc = posts[:, :, idx]
    idx = np.argmin(np.abs(bins[1] - preferred))
    return np.roll(slc, idx, 1)


# assumes logs of probabilities -> we can add instead of multiplying
# responses of channels
# preferred: list of the preferred orientations of the corresponding channels
def accumChannels(bins, posts, responses, preferred):
    # assert len(responses) == len(preferred), "responses and orientation not same number!"
    acc = np.zeros(np.shape(posts[:, :, 0]))
    for i, r in enumerate(responses):
        acc = acc + postsGivenResp(bins, posts, r, preferred[i])
    return acc


# assumes log of posterior prob
def accumTime(bins, posts, responses, preferred, windowSize):
    assert np.shape(responses)[1] == len(preferred), "number of channels of responses does not agree"
    # for every time step accumulate over channels
    postAll = np.zeros([np.shape(responses)[0], np.shape(posts)[0], np.shape(posts)[1]])
    for idx, r in enumerate(responses):
        postAll[idx, :, :] = accumChannels(bins, posts, r, preferred)
    # rolling sum over all to accumulate over time
    cs = np.cumsum(postAll, 0)
    cs[windowSize:, :, :] = cs[windowSize:, :, :] - cs[:-windowSize, :, :]
    return cs


# for every time step, just says what the most probable stimulus is
# has the advantage that we can incrementally adjust our estimates and quantify how wrong we are
def inferNaive(bins, posts):
    stims = np.zeros([2, np.shape(posts)[0]])
    shp = np.shape(posts[0, :, :])
    for i, p in enumerate(posts):
        mx = np.argmax(p)
        stIdx = np.unravel_index(mx, shp)
        # stims[0, i] = bins[0][stIdx[0] - 1]
        stims[0, i] = bins[0][stIdx[0]]
        stims[1, i] = bins[1][stIdx[1]]
    return stims


# this assumes a certain permanence of the stimuli
# when the properties of the stimulus are recognized, it is assumed that the stimulus stays that way, until another
# stimulus is deemed "much" (dependent on threshold) more likely
# problem: the properties of a stimulus cannot be as well incrementally adjusted as more information comes in
def inferredStim(bins, posts, thresh=np.log(2)):
    stims = np.zeros([2, np.shape(posts)[0]])
    shp = np.shape(posts[0, :, :])
    fIdx = np.argmax(posts[0, :, :])
    curStimId = np.unravel_index(fIdx, shp)
    stims[0, 0] = bins[0][curStimId[0]]
    stims[1, 0] = bins[1][curStimId[1]]
    # curStim = stims[:, 0]
    for i, p in enumerate(posts[1:]):
        mx = np.amax(p)
        confOldStim = p[curStimId[0], curStimId[1]]
        if mx - confOldStim >= thresh:
            idx = np.argmax(p)
            curStimId = np.unravel_index(idx, shp)
            stims[0, 0] = bins[0][curStimId[0]]
            stims[1, 0] = bins[1][curStimId[1]]
        else:
            stims[:, i + 1] = stims[:, i]


def performanceIdcs(stims, infStim, bins):
    assert np.shape(stims) == np.shape(infStim)
    stimIds = stims * 0
    infStimIds = stimIds * 0

    # for future reference: make sure all arrays/matrices are numpy arrays. this is a
    # list of numpy arrays, so we can't do cool [0, :, np.newaxis] indexing
    bn = bins[0][:, np.newaxis]
    stimIds[0, :] = np.sum(bn <= stims[0, :], 0)
    infStimIds[0] = np.sum(bn <= infStim[0, :], 0)

    bn = bins[1][:, np.newaxis]
    stimIds[1, :] = np.sum(bn <= stims[1, :], 0)
    infStimIds[1, :] = np.sum(bn <= infStim[1, :], 0)

    print("abs difference in idxs", np.sum(np.abs(stimIds - infStimIds)))
    print("difference in idxs", np.sum(stimIds - infStimIds))
    print("difference in idxs, cont only", np.sum(stimIds[0] - infStimIds[0]))
    print("difference in idxs, ori only", np.sum(stimIds[1] - infStimIds[1]))
    print("abs difference in idxs, cont only", np.sum(np.abs(stimIds[0] - infStimIds[0])))
    print("abs difference in idxs, ori only", np.sum(np.abs(stimIds[1] - infStimIds[1])))
    # orientation is overestimated about as much as underestimated
    # contrast is ALWAYS overestimated
    # absolute of incorrect of orientation and contrast seems comparable
    corrects = np.abs(stimIds - infStimIds <= 1)

    return corrects


def accuracy(stims, infStim, conZero=1):
    oridiffs = stims[0] - infStim[0]
    oriPerf = (-np.cos(oridiffs) + 1) / 2
    conPerf = np.abs(expCDF(stims[1], conZero) - expCDF(infStim[1], conZero))
    return conPerf, oriPerf


# mean of transformed/normalized differences
def performanceDiff(stims, infStim, durs, conZero=1, conRes=12):
    conspace = np.logspace(np.log10(np.amin(stims[0])), np.log10(np.amax(stims[0])), conRes+1)
    conPerf, oriPerf = accuracy(stims, infStim, conZero)
    fDur = np.zeros([2, int(np.amax(durs))])
    for d in range(int(np.amax(durs))):
        fDur[0, d] = myMean(conPerf[durs == d + 1])
        fDur[1, d] = myMean(oriPerf[durs == d + 1])

    fCon = np.zeros([2, conRes])
    for idx, c in enumerate(conspace[:-1]):
        select = (stims[0] >= c) & (stims[0] < conspace[idx + 1])
        fCon[0, idx] = myMean(conPerf[select])
        fCon[1, idx] = myMean(oriPerf[select])

    return fCon, fDur, conspace


def performanceSTD(stims, infStim, durs, conRes=12):
    diff = stims - infStim
    diff[1] = np.arccos(np.cos(diff[1]))
    # dependent on object duration
    fDur = np.zeros([2, int(np.amax(durs))])
    for d in range(int(np.amax(durs))):
        fDur[0, d] = myStd(diff[0][durs == d + 1])
        fDur[1, d] = myStd(diff[1][durs == d + 1])

    fCon = np.zeros([2, conRes])
    conspace = np.logspace(np.log10(np.amin(stims[0])), np.log10(np.amax(stims[0])), conRes+1)
    for idx, c in enumerate(conspace[:-1]):
        select = (stims[0] >= c) & (stims[0] < conspace[idx + 1])
        fCon[0, idx] = myStd(diff[0][select])
        fCon[1, idx] = myStd(diff[1][select])

    return fCon, fDur, conspace


def probLatency(stims, infStim, durs, conRes=12, thresh=0.1, conZero=1, durRes=12):
    ds, ss = cutDownDurs(durs)
    ds = np.array(ds)
    conAcc, oriAcc = accuracy(stims, infStim, conZero)
    conSingles = stims[0][ss]
    # --> go stimulus by stimulus
    # increment latency counter every step
    # if at some point inference is correct, set detected to True and latency to latency counter
    # otherwise (all inferences wrong): detected = False, latency = max (=stimulus length)
    results = [[False] * (len(ss)-1), [0] * (len(ss)-1)]
    for idxs, idxb in enumerate(ss[:-1]):
        ca = conAcc[idxb:ss[idxs + 1]]
        oa = oriAcc[idxb:ss[idxs + 1]]
        for l in range(ds[idxs]):
            if ca[l] <= thresh and oa[l] <= thresh:
                results[0][idxs] = True
                results[1][idxs] = l
                break
        else:
            results[1][idxs] = l
    # leave out last stimulus, because it's probably not full length, therefore distorting results
    # basically "histogramize" accumulate over the bins, make mean
    durspace = np.arange(1, np.amax(ds) + 1, ((np.amax(ds)-1) // (durRes-1)) + 1)
    fDur = np.zeros([2, len(durspace)])
    for idx, d in enumerate(durspace[:-1]):
        select = (ds[:-1] >= d) & (ds[:-1] < durspace[idx + 1])
        fDur[0, idx] = myMean(np.array(results[0])[select])
        fDur[1, idx] = myMean(np.array(results[1])[select])

    fCon = np.zeros([2, conRes])
    conspace = np.logspace(np.log10(np.amin(conSingles)), np.log10(np.amax(conSingles)), conRes+1)
    for idx, c in enumerate(conspace[:-1]):
        select = (conSingles[:-1] >= c) & (conSingles[:-1] < conspace[idx + 1])
        fCon[0, idx] = myMean(np.array(results[0])[select])
        fCon[1, idx] = myMean(np.array(results[1])[select])
    # probability top, latency bottom
    return fCon, fDur, conspace, durspace


def expCDF(x, lmb=1):
    return 1 - np.exp(-lmb * x)


def myStd(x):
    if len(x) == 0:
        return 0
    else:
        return np.nanstd(x)


def myMean(x):
    if len(x) == 0:
        return 0
    else:
        return np.nanmean(x)


def cutDownDurs(durs):
    ds = [durs[0]]
    selects = [0]
    while selects[-1] + ds[-1] < len(durs):
        selects.append(selects[-1] + ds[-1])
        ds.append(durs[selects[-1]])
    return ds, selects

