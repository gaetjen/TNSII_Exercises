import numpy as np


# give the corresponding posterior distribution of stimulus parameters, given the response
def postsGivenResp(bins, posts, response, preferred=0):
    assert 0 <= preferred <= (2 * np.pi), "preferred should be in range 0, 2π"
    idx = np.argmin(np.abs(bins[2] - response))

    slc = posts[:, :, idx]
    # tried this, so that not almost all values lead to an underestimated response makes virtually no difference
    # idx = np.argmin(np.abs(np.log(bins[1]) - np.log(preferred)))
    idx = np.argmin(np.abs(bins[1] - preferred))
    return np.roll(slc, idx, 1)


# assumes logs of probabilities -> we can add instead of multiplying
# responses of channels
# preferred: list of the preferred orientations of the corresponding channels
def accumChannels(bins, posts, responses, preferred):
    assert len(responses) == len(preferred), "responses and orientation not same number!"
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


def performanceDiff(stims, infStim, conZero=1):
    oridiffs = stims[0] - infStim[0]
    oriPerf = (-np.cos(oridiffs) + 1) / 2
    conPerf = np.abs(expCDF(stims[1], conZero) - expCDF(infStim[1], conZero))
    return oriPerf + conPerf, oriPerf, conPerf


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


def expCDF(x, lmb=1):
    return 1 - np.exp(-lmb * x)


def myStd(x):
    if len(x) == 0:
        return 0
    else:
        return np.nanstd(x)
