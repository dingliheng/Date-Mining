import numpy as np
from numpy import linalg as la
import sys
import math
import time

#from LossFunctions import write_metrics_given_support

# handle sparsity

from scipy.sparse.linalg import spsolve
import scipy.sparse as ss

""" ARGS
         j : index into I \otimes XtX
         d : j is an index into I \otimes XtX. I is identity of size d*d
         already_sel_dimensions : array containing indices in space of I\otimes XtX that are already selected
"""

def get_candidate_XtXvector(already_sel_dimensions, XtX, d, j):
    p = np.shape(XtX)[0]
    index_into_XtX = j %p
    num_blocks_of_zeros_before = int(j)/int(p) # how many blocks (size p*p) before XtX starts in j^{th} row of (I \otimes XtX)

    # j^{th} row of (I \otimes XtX) has zeros through indices [0... (num_blocks_of_zeros_before*p - 1) ], then row index_into_XtX of XtX, and then all zeros again
    non_zero_start = num_blocks_of_zeros_before*p
    non_zero_end = non_zero_start+ p - 1

    result = np.array([])
    for prev_sel_index in already_sel_dimensions:
        if(j == prev_sel_index ):
            print "ERROR: duplicate index selection!"
            exit(1)
        if(prev_sel_index >= non_zero_start and prev_sel_index <= non_zero_end):
            result = np.append(result, XtX[index_into_XtX, prev_sel_index - non_zero_start] )
        else:
            result = np.append(result, 0)

    return(result)

""" ARGS
         j : index into C \otimes I
         p : j is an index into C \otimes I. I is identity of size p*p
         already_sel_dimensions : array containing indices in space of I\otimes XtX that are already selected
"""
def get_candidate_Cvector(already_sel_dimensions, C, p, j):
    d = np.shape(C)[0]
    index_into_C  = int(j)/int(p)
    offset = j % p # offset is index of C entry C[index_into_C,0] in j^{th} row of C \otimes I
    print "offset = %d, index into c = %d" %(offset, index_into_C)
    # C \otimes I has first of C (potentially nonzero) at index=offset in row j.
    # in general entries C[index_into_C, j] are at offset + j * p

    result = np.array([])
    for prev_sel_index in already_sel_dimensions:
        if(j == prev_sel_index ):
            print "ERROR: duplicate index selection!"
            exit(1)
        if(int(prev_sel_index - offset) % p == 0 ):
            quotient = int(prev_sel_index - offset)/p
            print prev_sel_index, quotient, offset + quotient*p
            result = np.append(result, C[index_into_C, quotient])
        else:
            result = np.append(result, 0)
    return(result)

def main():
    XtX = np.matrix([[2,3,4],[3,10,6],[4,6,11]])
    already_sel_dimensions = np.array([4,2])
    print "startasda"
    print get_candidate_Cvector(already_sel_dimensions,C=XtX, p=4,j=5)
    pass


"""
    ARGS :    C = precision matrix of prior. size = d*d
              X = parameter matrix. size = n * p, p is NOT sparsity, it is size of lower dimension to be projected on for PCA
              T = observed matrix. size = n * d
              k = sparsity
              noiseVar = variance of error
"""
def solvePosteriorPrecisionMatrixVariateSparseGreedy(X, C, k, T, noiseVar, opt_dict, debug =0):
    #if( not ss.isspmatrix_csc(C)):
    #print "Error C is not csc sparse matrix"

    d = np.shape(C)[0]
    X = np.matrix(X)
    n = np.shape(X)[0]
    p = np.shape(X)[1]

    if(debug > 1):
        print " C has size " + str(np.shape(C))
        print " X has size " + str(np.shape(X))
        print " Sparsity = k" + str(k)

    # 1st selection
    argmax = -1
    maxx = -sys.float_info.max
    best_r = -1
    prev_maxx=0.0

    diag_xTx = np.array([0.0] * k)
    for ii in range(k):
        diag_xTx[ii] = np.dot(X[:, ii].T, X[:, ii])[0,0]

    XtX = X.T * X

    for j in range(p*d):
        index_into_XtX_term =  j % p  # index into I \otimes X^T X
        index_into_C_term = int(j)/int(p) # index into C \otimes I
        kk = C[index_into_C_term,index_into_C_term] + diag_xTx[index_into_XtX_term]/ noiseVar
        r = np.dot(  X[:, index_into_XtX_term].T, T[:,index_into_C_term]   )              # r = vec(X^T T). r[j] = dot( X[:, j%p],  T[:, j/p]  )
        r = r[0,0]

        f_trace_term_decider = r*r/kk

        f  = f_trace_term_decider - math.log(kk)
        if( f >  maxx):
            maxx = f
            argmax = j

    index_into_XtX_term =  argmax % p  # index into I \otimes X^T X
    index_into_C_term = int(argmax)/int(p) # index into C \otimes I

    if( debug > 0):
        print "1st choice : "+str(argmax)+" index_in_XtX= " +str(index_into_XtX_term) + " index_into_C_term= " +str(index_into_C_term) +" with f= "+str(maxx)

    already_built_pspinv = np.matrix(C[index_into_C_term,index_into_C_term]+diag_xTx[index_into_XtX_term]/noiseVar) # atmost k*k; dense, greedily built this matrix
    inv_already_built_pspinv = la.inv(already_built_pspinv) # size 1. just does reciprocal

    r = np.dot(  X[:, index_into_XtX_term].T, T[:,index_into_C_term]   )

    already_selected_dimensions = {argmax:1} #dict of size atmost k, exactly size k at the end. This'll be returned at the end
    prev_selected = np.array([argmax])
    r1 = np.array(r) # subvector of r on selected dimensions

    if( debug > 0):
        print "1st choice : "+str(argmax)+" with f="+str(maxx)

    # after 1st selection
    for i in range(1,k):
        maxx = -sys.float_info.max
        argmax = -1
#        prev_selected = already_selected_dimensions.keys()

        for j in range(d):
            if(not already_selected_dimensions.has_key(j)):
                # trace term

                candidate_XtX_column = get_candidate_XtXvector(already_sel_dimensions=prev_selected, XtX=XtX, d=d, j=j) # n *1
                candidate_C_row = get_candidate_Cvector(already_sel_dimensions=prev_selected, C=C, p=p, j=j) # 1* (i-1) size, row or col doesnt matter

                b  = np.matrix(candidate_C_row + candidate_XtX_column)
                w = np.dot(inv_already_built_pspinv, b.T)

                index_into_XtX_term =  j % p  # index into I \otimes X^T X
                index_into_C_term = int(j)/int(p) # index into C \otimes I
                kk = C[index_into_C_term,index_into_C_term] + XtX[index_into_XtX_term, index_into_XtX_term]/ noiseVar - np.dot(b,w)[0,0]

                r = np.dot(  X[:, index_into_XtX_term].T, T[:,index_into_C_term]   )              # r = vec(X^T T). r[j] = dot( X[:, j%p],  T[:, j/p]  )
                r = r[0,0]
                tt = np.dot(r1, w) -r
                f_trace_term_decider = tt*tt/kk

                # determinant term
                logdetPSPinv =   math.log(abs(kk))  # kk should always be positive, abs here is protect from any underflows

                f = f_trace_term_decider - logdetPSPinv

                if( f >  maxx):
                    maxx = f
                    argmax = j

#print "After iter %d, max=%f, argmax=%d" %(i,maxx, argmax)
        diff_maxx = maxx - prev_maxx
        prev_maxx = maxx

        # concatenate chosen columns into PSPinv
        candidate_XtX_column = get_candidate_XtXvector(already_sel_dimensions=prev_selected, XtX=XtX, d=d, j=argmax) # n *1
        candidate_C_row = get_candidate_Cvector(already_sel_dimensions=prev_selected, C=C, p=p, j=argmax) # 1* (i-1) size, row or col doesnt matter

        b  = np.matrix(candidate_C_row + candidate_XtX_column)
        w = np.dot(inv_already_built_pspinv, b.T)

        index_into_XtX_term =  argmax % p  # index into I \otimes X^T X
        index_into_C_term = int(argmax)/int(p) # index into C \otimes I
        kk = C[index_into_C_term,index_into_C_term] + XtX[index_into_XtX_term, index_into_XtX_term]/ noiseVar - np.dot(b,w)[0,0]

        r = np.dot(  X[:, index_into_XtX_term].T, T[:,index_into_C_term]   )              # r = vec(X^T T). r[j] = dot( X[:, j%p],  T[:, j/p]  )
        r = r[0,0]

        inv_already_built_pspinv = inv_already_built_pspinv + (1.0/kk) * np.matrix(np.outer(w, w))
        inv_already_built_pspinv = np.concatenate((inv_already_built_pspinv,  (-1.0/kk)*w.T),axis=0)
        wk = np.concatenate(((-1.0/kk)*w.T, np.matrix(   np.array( [1/kk])    )), axis=1)
        inv_already_built_pspinv = np.concatenate((inv_already_built_pspinv, wk.T),axis=1)

        prev_logdetPSPinv = logdetPSPinv
        r1 = np.append(r1, r)
        already_selected_dimensions[argmax] =1
        prev_selected = np.append(prev_selected, argmax)

#        write_metrics_given_support(already_selected_dimensions.keys(),k=i, chosen_this_iter=argmax, fmax=maxx, diff_maxx=diff_maxx, C = C,rtrain=r, noiseVar=noiseVar, opt_dict=opt_dict, is_p_fullvector=0)

    pmu = np.array(np.dot(inv_already_built_pspinv, r1)).ravel()
    return(prev_selected, pmu)

    pass


"""
    ARGS :    C = precision matrix of prior. size = d*d
              X = features matrix. size = n *d
              r = (1/noiseVar) X^T y
              diag_xTx = size n. as name says
              mu = mean
              k = sparsity
              noiseVar = variance of error
"""
def solvePosteriorPrecisionSparseGreedy(X, C, k, r, noiseVar, diag_xTx, opt_dict, debug =0):
    #if( not ss.isspmatrix_csc(C)):
    #print "Error C is not csc sparse matrix"

    d = np.shape(C)[0]
    X = np.matrix(X)
    n = np.shape(X)[0]
    r = np.array(r)
#    p = np.array([0] * d ) # selected rows/cols. instead of 0,1

    if(debug > 1):
        print " C has size " + str(np.shape(C))
        print " X has size " + str(np.shape(X))
        print " Sparsity = k" + str(k)

    # 1st selection
    argmax = -1
    maxx = -sys.float_info.max
    prev_maxx=0.0


    start_iter_time = time.clock()
    for j in range(d):
        kk = C[j,j] + diag_xTx[j]/ noiseVar
        f_trace_term_decider = r[j] * r[j]/kk
        f  = f_trace_term_decider - math.log(kk)
        if( f >  maxx):
            maxx = f
            argmax = j

    end_iter_time = time.clock()
#print "After iter %d, max=%f, argmax=%d time_taken= %f" %(0,maxx, argmax, end_iter_time - start_iter_time)

    already_built_pspinv = np.matrix(C[argmax,argmax]+diag_xTx[j]/noiseVar) # atmost k*k; dense, greedily built this matrix
    inv_already_built_pspinv = la.inv(already_built_pspinv) # size 1. just does reciprocal
    already_selected_X = np.matrix(X[:,argmax]) # dense, atmost n*k. X is dense, it'll be faster if we store selected X rather than creating submatrix from full X everytime.

    already_selected_dimensions = {argmax:1} #dict of size atmost k, exactly size k at the end. This'll be returned at the end
    r1 = np.array([r[argmax]]) # subvector of r on selected dimensions
    prev_selected=np.array([argmax])

    prev_logdetPSPinv = 0.0
    diff_maxx = maxx-prev_maxx
    prev_maxx = maxx
    end_bookkeeping_time = time.clock()
    #print "Iter %d bookkeeping time= %f" %(0, end_bookkeeping_time  - end_iter_time)

    #write_metrics_given_support(already_selected_dimensions.keys(),k=0, chosen_this_iter=argmax, fmax=maxx, diff_maxx=diff_maxx, C = C,rtrain=r, noiseVar=noiseVar, opt_dict=opt_dict, is_p_fullvector=0)

    # after 1st selection
    for i in range(1,k):
        maxx = -sys.float_info.max
        argmax = -1
#        prev_selected = already_selected_dimensions.keys()
        start_iter_time = time.clock()

        for j in range(d):

            if(not already_selected_dimensions.has_key(j)):
                # trace term

                candidate_X_column = X[:, j] # n *1
                candidate_C_row = C[j,prev_selected] # 1* (i-1) size, row or col doesnt matter
                newrow_contrib_fromx =np.dot( candidate_X_column.T, already_selected_X)/noiseVar # 1 * (i-1)
                b  = candidate_C_row + newrow_contrib_fromx
#                 w = la.solve(already_built_pspinv, b.T)
#                 w = np.dot(already_built_pspinv.I, b.T)
                w = np.dot(inv_already_built_pspinv, b.T)
                kk = C[j,j] + diag_xTx[j]/noiseVar - np.dot(b,w)

                tt = np.dot(r1, w) -r[j]
                f_trace_term_decider = tt*tt/kk

                # determinant term
#                 logdetPSPinv = prev_logdetPSPinv +  math.log(abs(kk))  # kk should always be positive, abs here is protect from any underflows
                logdetPSPinv =   math.log(abs(kk))  # kk should always be positive, abs here is protect from any underflows

                f = f_trace_term_decider - logdetPSPinv

                if( f >  maxx):
                    maxx = f
                    argmax = j
#        print maxx, argmax
        # book keeping for next iter
        end_iter_time = time.clock()
       #print "After iter %d, max=%f, argmax=%d" %(i,maxx, argmax)
#print "After iter %d, max=%f, argmax=%d time_taken= %f" %(i,maxx, argmax, end_iter_time - start_iter_time)

        diff_maxx = maxx - prev_maxx
        prev_maxx = maxx


        # concatenate chosen columns into PSPinv
        candidate_X_column = X[:, argmax] # n *1
        candidate_C_row = C[argmax,prev_selected] # 1* (i-1) size, row or col doesnt matter
        newrow_contrib_fromx =np.dot( candidate_X_column.T, already_selected_X)/noiseVar # 1 * (i-1)
        b  = candidate_C_row + newrow_contrib_fromx # 1*(i-1)
        w = np.dot(inv_already_built_pspinv, b.T)
        kk = C[argmax,argmax] + diag_xTx[argmax]/noiseVar - np.dot(b,w)
        kk = kk[0,0] # kk is a matrix with single element, extract it to be used as scalar.

        inv_already_built_pspinv = inv_already_built_pspinv + (1.0/kk) * np.matrix(np.outer(w, w))
        inv_already_built_pspinv = np.concatenate((inv_already_built_pspinv,  (-1.0/kk)*w.T),axis=0)
        wk = np.concatenate(((-1.0/kk)*w.T, np.matrix(   np.array( [1/kk])    )), axis=1)
        inv_already_built_pspinv = np.concatenate((inv_already_built_pspinv, wk.T),axis=1)

        prev_logdetPSPinv = logdetPSPinv
        r1 = np.append(r1, r[argmax])
        already_selected_dimensions[argmax] =1
        already_selected_X = np.concatenate((already_selected_X,X[:,argmax]), axis=1)
        prev_selected = np.append(prev_selected, argmax)

        end_bookkeeping_time = time.clock()
    #print "Iter %d bookkeeping time= %f" %(i, end_bookkeeping_time  - end_iter_time)

        #write_metrics_given_support(already_selected_dimensions.keys(),k=i, chosen_this_iter=argmax, fmax=maxx, diff_maxx=diff_maxx, C = C,rtrain=r, noiseVar=noiseVar, opt_dict=opt_dict, is_p_fullvector=0)

    pmu = np.array(np.dot(inv_already_built_pspinv, r1)).ravel()
    return(prev_selected, pmu)

"""
    ARGS :    C = precision matrix of prior. size = d*d
              X = features matrix. size = n *d
              r = (1/noiseVar) X^T y
              mu = mean
              k = sparsity
              noiseVar = variance of error
"""
# Version 1 - everything basic - no smart matrix building
def solvePosteriorPrecisionGreedy(X, C, k, r, noiseVar, opt_dict, debug =1):
#    C = np.matrix(C)
    d = np.shape(C)[0]
    X = np.matrix(X)
    r = np.array(r)
    p = np.array([0] * d ) # selected rows/cols

    if(debug >1):
        print " C has size " + str(np.shape(C))
        print " X has size " + str(np.shape(X))
        print " Sparsity = k" + str(k)
    prev_maxx = 0

    for i in range(k):
        maxx = -sys.float_info.max
        argmax = -1
        where1 = np.where(p == 1)[0]
        for j in np.where(p==0)[0]:
            pp = np.append(where1, j)
            submatC = C[pp,:][:,pp]
            submatX = X[:, pp]
            submatXtX = submatX.T * submatX
            PSPinv = submatC + submatXtX/noiseVar
            sub_r = r[pp]
            PSPr = la.solve(PSPinv, sub_r)
            rPSPr = np.dot(sub_r, PSPr)

            f =  rPSPr - math.log(la.det(PSPinv))
            if( (j==350 or j==0) and i==0):
                print "j=%d, noiseVar=%f, C[j,j]=%f, diag_xTx[j]=%f, r[j]=%f, kk=%f, f_trace_term_decider=%f, -log(kk)=%f, f=%f"    %(j, noiseVar, C[j,j], submatXtX, r[j], PSPinv, rPSPr, -math.log(PSPinv),f )

            # DEBUG{
#             if( j == 0 or j==1):
#                 print "j=%d, f= %f" %(j, f)
#             #}

            if( f >  maxx):
                maxx = f
                argmax = j
    #if(debug > 0):
        #print "After iter %d, max=%f, argmax=%d" %(i,maxx, argmax)
        p[argmax]=1
        diff_maxx = maxx - prev_maxx
        prev_maxx= maxx
        write_metrics_given_support(p,k=i, chosen_this_iter=argmax, fmax=maxx, diff_maxx=diff_maxx, C = C,rtrain=r, noiseVar=noiseVar, opt_dict=opt_dict)

    return(p)





"""
    ARGS :    C = precision matrix of prior. size = d*d
              X = features matrix. size = n *d
              r = (1/noiseVar) X^T y
              mu = mean
              k = sparsity
              noiseVar = variance of error
"""
# Version 1 - everything basic - no smart matrix building
def solvePosteriorPrecisionGreedyGroup(X, C, k, r, noiseVar, opt_dict, debug =1, allgroups=None):
#    C = np.matrix(C)
    d = np.shape(C)[0]
    X = np.matrix(X)
    r = np.array(r)
    ngroups = len(allgroups)

    p = np.array([0] * ngroups) # selected groups
    selected_dim = np.array([0]*d)

    if(debug >1):
        print " C has size " + str(np.shape(C))
        print " X has size " + str(np.shape(X))
        print " Sparsity = k" + str(k)
    prev_maxx = 0

    for i in range(k):
        maxx = -sys.float_info.max
        argmax = -1

        dim_where1 = np.where(selected_dim == 1)[0]

        for j in np.where(p==0)[0]: # group
            pp = np.append(dim_where1, np.array(allgroups[j]))

            submatC = C[pp,:][:,pp]
            submatX = X[:, pp]
            submatXtX = submatX.T * submatX
            PSPinv = submatC + submatXtX/noiseVar
            sub_r = r[pp]
            PSPr = la.solve(PSPinv, sub_r)
            rPSPr = np.dot(sub_r, PSPr)

            f = rPSPr - math.log(la.det(PSPinv))
            if(f >  maxx):
                maxx = f
                argmax = j
    #if(debug > 0):
        #print "After iter %d, max=%f, argmax=%d" %(i,maxx, argmax)
        p[argmax]=1
        selected_dim[np.array(allgroups[argmax])] = 1
        diff_maxx = maxx - prev_maxx
        prev_maxx= maxx
        write_metrics_given_support(selected_dim, k=i, chosen_this_iter=argmax, fmax=maxx, diff_maxx=diff_maxx, C = C,rtrain=r, noiseVar=noiseVar, opt_dict=opt_dict)

    return(selected_dim)

""" PURPOSE : solve min_P  - ln det | PCP |  + trace(PCP Pinv(C)P), where P is diag matrix that has 1s or 0s on its diagonal, total number of ones = k
    ARGS:    C -> (numpy matrix) see PURPOSE
             k -> (scalar) number of 1s allowed.
    RETURN:  p, P = diag(p), where P as defined in PURPOSE
"""
def solvePriorGreedy(C, k):
    print "this method is wrong, exitting!"
    exit(1)
    C = np.matrix(C)
    invC = C.I
    p = np.array([0] * np.shape(C)[0]) # selected rows/cols
    #p = []
    n = np.shape(C)[0]
    if( k >= n):
        print "k should be < n"
        exit(1)
    for i in range(k):
        minn = 999999
        argmin = -1
        where1 = np.where(p == 1)[0]
        for j in np.where(p==0)[0]:
            pp = np.append(where1, j)
            submatC = C[pp,:][:,pp]
            submatinvC = invC[pp,:][:,pp]
            f= np.trace(submatinvC * submatC ) - np.log( la.det(submatC))
#            print "j = %d, f=%f" %(j,f)
            if( f < minn):
                minn = f
                argmin = j
#        print "argmin chosen: %d" %argmin
        p[argmin]=1
    return(p)

"""NOTE : posterior is actually general version of prior
"""
""" PURPOSE : solve min_P (mu - Pmu)*invC*(mu - Pmu) -  ln det | PCP |  + trace(PCP Pinv(C)P), where P is diag matrix that has 1s or 0s on its diagonal, total number of ones = k
    ARGS:    C -> (numpy matrix) see PURPOSE
             k -> (scalar) number of 1s allowed.
    RETURN:  p, P = diag(p), where P as defined in PURPOSE
"""
def solvePosteriorGreedy(mu, C, k):
    print "this method is wrong, exitting!"
    exit(1)
    C = np.matrix(C)
    invC = C.I
    p = [0] * np.shape(C)[0] # selected rows/cols
    n = np.shape(C)[0]
    if( k >= n):
        print "k should be < n"
        exit(1)
    for i in range(k):
        minn = 999999
        argmin = -1
        where1 = np.where(p == 1)[0]
        for j in np.where(p==0)[0]:
            pp = np.append(where1, j)
            submatC = C[pp,:][:,pp]
            submatinvC = invC[pp,:][:,pp]
            p1 = p
            p1[j]=1
            mu1 = np.dot(np.diag(p1), mu)
            f = np.dot(np.dot((mu - mu1),invC), (mu - mu1))[0,0] + np.trace(submatinvC * submatC ) - np.log( la.det(submatC))
            if( f < minn):
                minn = f
                argmin = j
        p[argmin]=1
    return(p)

""" PURPOSE : solve min_P y^T *inv{E}*y -  ln det | E |, where E = PCP + scaledXXt, scaledXXt = (1/\sigma^2) XX^T, \sigma^2= variance(y), P is diag matrix with 1s or 0s at its diagonal, total num 1s =k
    ARGS:    C -> (numpy matrix) see PURPOSE
             k -> (scalar) number of 1s allowed.
    RETURN:  p, P = diag(p), where P as defined in PURPOSE
"""
def solveMLGreedy(X, y, C, noiseVar, k):
    C = np.matrix(C)
    X = np.matrix(X)
    y= np.array(y)
    n = len(y)
    p = np.array([0] * np.shape(C)[0])
    print k

    for i in range(k):
        maxx = -99999
        argmax = -1
        for j in np.where(p==0)[0]:
            pp = p
            pp[j] = 1
            E = X * np.diag(pp) * C * np.diag(pp) * X.T + noiseVar * np.eye(n)
            invE = E.I

            f = -1 * np.dot(np.dot(invE, y),y)[0,0] + np.log( la.det(invE))
            print f
            if(f > maxx):
                maxx= f
                argmax = j
        p[argmax] = 1
    print p
    return(p)

def solveMLFull(X, y, C, noiseVar, k):
    C = np.matrix(C)
    X = np.matrix(X)
    n = len(y)
    scores_dict = {}
    # broken
    print "broken function"
    exit(1)

if __name__ == "__main__":
    main()

