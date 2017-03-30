import numpy as np
from sklearn.linear_model import Lasso
from numpy import linalg as la
from sklearn.linear_model import ARDRegression
import time

""" Fit ard by reweighted lasso as specified in Wipf et. al. A New View of Automatic Relevance Determination
    ARGS: Xtrain -> training features
          ytrain -> training y
          noiseVar -> ytrain ~ Normal ( Xtrain * w, noiseVar)
          eps -> convergence slack
          max_iters -> max number of iters to run, loop terminates if convergence before max_iters

    RETURNS :    intercept -> mean(ytrain), ytrain is made 0 mean before fitting
                 w -> fitted weight vector
                 gamma -> vector of fitted regularization on each dimension
"""

def iterative_ard(Xtrain, ytrain, noiseVar, eps = 0.1, max_iters = 20, debug=1):
    Xtrain  = np.array(Xtrain)
    ytrain = np.array(ytrain)

    intercept = np.mean(ytrain)
    ytrain  = ytrain - intercept

    n,d = np.shape(Xtrain)
    tnoise = 2*noiseVar

    if(debug > 1):
        print "shape of X = "+ str(n)+" " +str(d)
        print "shape of y = " + str(np.shape(ytrain))

    z = np.ones(d) # init


    diff = 10
    i = 0
    old_gamma = z # to check convergence
    gamma = np.ones(d)

    while( i < max_iters)  and  (diff > eps):
        i = i + 1

 #       rewts = np.sqrt(z)

        # step 1 re-weighted lasso
#        weighted_x = Xtrain / rewts #  x[:,i] * z[i]

        laso = Lasso(alpha = tnoise, max_iter= 100000)
        laso_fit = laso.fit(Xtrain/(np.sqrt(z)), ytrain)
        gamma = np.abs(laso.coef_) / z # gamma_i = (1/sqrt(z_i)) * actual_regularized_w = (1/sqrt(z_i))* (1/sqrt(z_i)) * laso_fit.coef_[i]
        # step 1 done
        # step 2, update z

#        Sigma_y = noiseVar * np.eye(n) + np.dot(Xtrain, np.dot(np.diag(gamma), np.transpose(Xtrain) )) # WONT work if d is very large, coz diag(gamma) blows up
        Sigma_y = noiseVar * np.eye(n) + np.dot(Xtrain,  np.transpose(Xtrain * gamma) )

        #sigma_rank = la.matrix_rank(Sigma_y)
#         if(debug > 0 ):
#             print "rank of Sigma_y = %d" %(sigma_rank)

#        if(sigma_rank == n):
        if(0==1): # never run actual inverse
            z = np.dot(la.inv(Sigma_y), Xtrain)
        elif(1==2): # older slower code
            for j in range(d):
                SigmaInvX = la.solve(Sigma_y, Xtrain[:,j])
                z[j] = np.dot(Xtrain[:,j], SigmaInvX)
                if(z[j] <= 0):
                    if(debug > 0):
                        print "Zj <= 0 for j = %d, Zj = %f" %(j, z[j])
                    z[j] = 0.0001
        else:
            InvsigmaX = la.solve(Sigma_y, Xtrain)
            XtInvsigmaX = np.array(Xtrain) * np.array(InvsigmaX) # hadamard product
            z = np.array(np.sum(XtInvsigmaX, 0)).ravel()
            qq = np.where(z <= 0)[0]
            z[qq] = 0.0001
        # step 2 done

        # book keeping
        diff = la.norm(gamma - old_gamma, 2)
        old_gamma = np.array(gamma)
        if(debug > 0):
            print "Iter %d change=%f" %(i, diff)

    # compute w to return

 #   Sigma_y = noiseVar * np.eye(n) + np.dot(Xtrain, np.dot(np.diag(gamma), np.transpose(Xtrain) ))
    Sigma_y = noiseVar * np.eye(n) + np.dot(Xtrain,  np.transpose(Xtrain * gamma) )
    SigmaInvY = la.solve(Sigma_y, ytrain)
    XTSigmaInvY = np.dot(np.transpose(Xtrain), SigmaInvY)
#    w = np.dot(np.diag(gamma),XTSigmaInvY)
    w = gamma * XTSigmaInvY
    return(intercept,w, gamma)

def main_bak():
    # trial
    noiseVar = 0.01
    n=500
    d= 10

    x = np.random.normal(0,1, size = d*n ).reshape((n,d))
    w = np.random.normal(10,1,size=d)
    y = np.dot(x,w) + np.random.normal(0,noiseVar, size = n )

    t1 = time.time()

    print "Running iterative ard"
    (witer,gamma) = iterative_ard(Xtrain = x,ytrain =y,noiseVar = noiseVar)
    t2 = time.time(
                   )
    print "Running scikit ARD"
    ard = ARDRegression(compute_score=True)
    ard.fit(x, y)
    t3 = time.time()

    print "Time taken "
    print "Iterative:" + str(t2-t1)
    print "scikit ard:" + str(t3-t2)

    print "ALL W :"
    print witer
    print ard.coef_
    print w

def main():

    pass

if __name__ == "__main__":
    main()
