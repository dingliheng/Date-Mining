import numpy as np
from Solver import solvePosteriorPrecisionSparseGreedy

def run_submodregression(Xtrain, ytrain, k, sigma=1):
    r = np.dot(np.transpose(Xtrain), ytrain)/sigma
    C= np.eye(np.shape(Xtrain)[1])

    diag_xTx = np.array([0.0] * np.shape(Xtrain)[1])
    for ii in range(len(diag_xTx)):
        diag_xTx[ii] = np.dot(Xtrain[:, ii], Xtrain[:, ii])
    #print "done"

    where1, pmu= solvePosteriorPrecisionSparseGreedy(X = Xtrain, C=C, k = k, r=r, noiseVar=sigma, diag_xTx = diag_xTx, debug=0,opt_dict=None)

    beta = np.zeros(np.shape(Xtrain)[1])
    beta[where1] = pmu
    return beta


