import numpy as np
# import numpy.matlib as ml
import random as rnd
import math
import probabilities as prob


def genStimuli(contrast, duration, time, deltaT):
    # GENSTIMULI generate sequence of stimuli
    #    contrast: mean contrast
    #    duration: mean duration
    #    time:     total time of stimuli to generate
    #    deltaT:   time resolution (step size)
    #    stimuli:  contrast (top row) and orientation (bottom row) over time
    stimuli = np.zeros([2, round(time/deltaT)])
    durations = np.zeros(round(time/deltaT), dtype=int)
    t = 0
    idx = 0
    while t <= time:
        dur = np.random.exponential(duration)
        t += dur
        idxO = idx
        idx = int(round(t / deltaT))
        if t > time:
            idx = len(durations)
        durations[idxO:idx] = idx - idxO

        con = np.random.exponential(contrast)
        ori = rnd.random() * 2 * math.pi
        stimuli[0, idxO:idx] = con
        stimuli[1, idxO:idx] = ori


    return stimuli, durations


def response(numChannels, stimuli, tuningWidth, sensitivity):
    # RESPONSE responses of channels to stimuli
    #    numChannels: number of channels
    #    stimuli:     stimuli over time, comprising contrast (top) and
    #    orientation (bottom)
    #    tuningWidth: tuning width for orientation
    #    sensitivity: channels' sensitivity
    #    returns: numStimuli * numChannels of responses
    preferred = np.linspace(0, 2*math.pi, numChannels, False)
    means = sensRep(preferred, stimuli, tuningWidth, sensitivity)
    responses = np.random.rayleigh(means / math.sqrt(math.pi / 2))
    # responses = means
    return responses, preferred


def sensRep(preferred, stimuli, tuningWidth, sensitivity):
    # SENSREP calculate sensory representation of stimuli (mean response, given orientation and contrast)
    #    preferred: preferred stimulus orientation of different channels
    #    stimuli: 2×t array of stimuli, contrast (top) and orientation (bottom)
    #    returns: t * len(pref) array of mean responses
    contrast = np.transpose(np.tile(stimuli[0, :], [len(preferred), 1]))
    orientation = np.transpose(np.tile(stimuli[1, :], [len(preferred), 1]))
    return prob.meanResp(contrast, orientation, preferred, tuningWidth, sensitivity)

