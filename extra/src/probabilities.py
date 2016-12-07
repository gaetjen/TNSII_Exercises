import numpy as np
import scipy.stats as stat


def pOri(theta=0):
    return 1 / (2 * np.pi)


def pOriLog(theta=0):
    return np.log(pOri())


def pCon(con, conZero):
    return np.exp(-con / conZero) / conZero


def pConLog(con, conZero=1):
    return np.log(pCon(con, conZero))


def meanResp(con, ori, oriPreferred, tuningWidth=1, sensitivity=1):
    return sensitivity * con * np.exp(2 * np.cos(tuningWidth * (oriPreferred - ori)))


def pRespGivenOriCon(resp, ori, con, oriPreferred, tuningWidth=1, sensitivity=1):
    mr = meanResp(con, ori, oriPreferred, tuningWidth, sensitivity)
    return np.pi * resp / (2 * mr * mr) * np.exp(-np.pi * resp * resp / (4 * mr * mr))


def pRespGivenOriConLog(resp, ori, con, oriPreferred, tuningWidth=1, sensitivity=1):
    return np.log(pRespGivenOriCon(resp, ori, con, oriPreferred, tuningWidth, sensitivity))


def pJointConOri(ori, con, conZero):
    return pOri(ori) * pCon(con, conZero)


def pJointConOriLog(ori, con, conZero):
    return pOriLog(ori) + pConLog(con, conZero)


def pJointAll(resp, ori, con, conZero, oriPreferred=0, tuningWidth=1, sensitivity=1):
    if con == 0 or resp == 0:
        return 0
    cond = pRespGivenOriCon(resp, ori, con, oriPreferred, tuningWidth, sensitivity)
    return cond * pJointConOri(ori, con, conZero)


def pJointAllLog(resp, ori, con, conZero, oriPreferred=0, tuningWidth=1, sensitivity=1):
    cond = pRespGivenOriConLog(resp, ori, con, oriPreferred, tuningWidth, sensitivity)
    return cond + pJointConOriLog(ori, con, conZero)


def inverseExponCDF(x, lmb=1):
    return - np.log(1 - x) / lmb
