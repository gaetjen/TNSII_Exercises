import numpy as np
import numpy.matlib as ml
import random as rnd
import math


def genStimuli(contrast, duration, time, deltaT):
    # GENSTIMULI generate sequence of stimuli
    #    contrast: mean contrast
    #    duration: mean duration
    #    time:     total time of stimuli to generate
    #    deltaT:   time resolution (step size)
    #    stimuli:  orientation (top row) and contrast (bottom row) over time
    stimuli = np.zeros([2, time/deltaT])
    t = 0
    while t <= time:
        dur = np.random.exponential(duration)
        con = np.random.exponential(contrast)
        ori = rnd.random() * 2 * math.pi
        idx = round(t / deltaT)
        stimuli[0, idx:] = ori
        stimuli[1, idx:] = con
        t += dur
    return stimuli


def response(numChannels, stimuli, tuningWidth, sensitivity):
    # RESPONSE responses of channels to stimuli
    #    numChannels: number of channels
    #    stimuli:     stimuli over time, comprising orientation (top) and
    #    contrast (bottom)
    #    tuningWidth: tuning width for orientation
    #    sensitivity: channels' sensitivity
    preferred = np.linspace(0, 2*math.pi, numChannels, False)
    means = sensRep(preferred, stimuli, tuningWidth, sensitivity)
    responses = np.random.rayleigh(means / math.sqrt(math.pi / 2))
    return responses


def sensRep(preferred, stimuli, tuningWidth, sensitivity):
    # SENSREP calculate sensory representation of stimuli
    #    preferred: preferred stimulus orientation of different channels
    #    stimuli: 2×t array of stimuli, orientation (top) and contrast (bottom)
    contrast = ml.repmat(stimuli[1, :], len(preferred), 1)
    orientation = ml.repmat(stimuli[0, :], len(preferred), 1)
    preferred = ml.repmat(preferred, np.size(stimuli, 1), 1)
    means = sensitivity * contrast * np.exp(2 * math.cos(tuningWidth * (preferred - orientation))) # circular gaussian tuning
    return means



