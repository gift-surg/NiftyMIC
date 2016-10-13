#!/usr/bin/env python
import numpy as np
import scipy.optimize, scipy.ndimage
import matplotlib.pyplot as plt
import time

def lsmr_with_init(A,b,x0):
    r0 = b - scipy.sparse.linalg.aslinearoperator(A).matvec(x0)
    deltax_pack = scipy.sparse.linalg.lsmr(A,r0)
    return x0 + deltax_pack[0]

def fgr_nnlsqr(A,b):
    # The Lipschitz constant should be the max eigenvalue of A^T.A
    # To avoid computing it we use the squared Frobenius norm of A
    # which is an upperbound
    #Anorm2 = np.dot(A.ravel(),A.ravel())

    Aop = scipy.sparse.linalg.aslinearoperator(A)
    Anorm2 = 0
    for i in range(0, A.shape[1]):
        x = np.zeros(A.shape[1])
        x[i] = 1
        y = Aop.matvec(x)
        Anorm2 = Anorm2 + np.dot(y,y)
    
    invL = 1.0/Anorm2

    #print np.linalg.norm(np.dot(np.transpose(A),A),2)

    #print 'Anorm2 =', Anorm2
    #print '1/L =', invL

    zero_init = True
    if zero_init:
        x = np.zeros(A.shape[1])
    else:
        x_pack =  scipy.sparse.linalg.lsmr(Aop,b)
        x = np.clip(x_pack[0], 0, np.inf )

    xold = x
    y = x
    theta = 1.0
    thetaold = 1.0

    tol = 1e-5
    max_iter = 500
    diff = np.inf
    iter = 0
    while ( diff>tol and iter<max_iter ):
        iter += 1

        thetaold = theta
        theta = 2.0/(iter+2)
        
        xold = x

        #print iter,x
        y = x + (theta*((1.0/thetaold)-1.0))*(x-xold)

        x = np.clip( y - invL*Aop.rmatvec( Aop.matvec(y)-b ), 0.0, np.inf )

        if ( iter % 50 ):
            # restart
            y = x
            x = np.clip( y - invL*Aop.rmatvec( Aop.matvec(y)-b ), 0.0, np.inf )

        diff = np.linalg.norm(x-xold)
        
        #print 'iter =', iter, np.linalg.norm(Aop.matvec(x)-b)
        
    return x

def mfista_nnlsqr(A,b):
    # The Lipschitz constant should be the max eigenvalue of A^T.A
    # To avoid computing it we use the squared Frobenius norm of A
    # which is an upperbound
    #Anorm2 = np.dot(A.ravel(),A.ravel())
    Aop = scipy.sparse.linalg.aslinearoperator(A)
    Anorm2 = 0
    for i in range(0, A.shape[1]):
        x = np.zeros(A.shape[1])
        x[i] = 1
        y = Aop.matvec(x)
        Anorm2 = Anorm2 + np.dot(y,y)
    #Anorm = scipy.sparse.linalg.norm(Aop,'fro')
    #Anorm2 = Anorm*Anorm
    L = Anorm2
    invhalfL = 2.0/L
    t = 1.0
    told = 1.0

    zero_init = False
    if zero_init:
        x = np.zeros(A.shape[1])
    else:
        x_pack =  scipy.sparse.linalg.lsmr(Aop,b)
        x = np.clip(x_pack[0], 0, np.inf )

    xold = x
    y = x
    z = x

    tol = 1e-5
    max_iter = 500
    diff = np.inf
    iter = 0
    while ( diff>tol and iter<max_iter ):
        iter += 1

        #if ( iter % 20 ):
        #    # restart
        #    y = x
        
        xold = x
        told = t
        z = np.clip( y - invhalfL*Aop.rmatvec( Aop.matvec(y)-b ), 0, np.inf )
        #print invhalfL* Aop.rmatvec( Aop.matvec(y)-b )
        #print z
        t = (1.0+np.sqrt(1.0+4.0*t*t))/2.0
        #print t
        tmp = Aop.matvec(z)-b
        Fz = np.dot(tmp.ravel(),tmp.ravel())
        tmp = Aop.matvec(x)-b
        Fx = np.dot(tmp.ravel(),tmp.ravel())
        #print Fx, Fz
        x = x if (Fx < Fz or np.isnan(Fz)) else z
        #print x
        y = x + (told/t)*(z-x) + ((told-1)/t)*(x-xold)

        diff = np.linalg.norm(x-xold)

        #print 'iter =', iter, np.linalg.norm(Aop.matvec(x)-b)

    return x
    

def pdhgmp_nnlsqr(A,b):
    # Algorithm 8 in
    # First Order Algorithms in Variational Image Processing
    # minimise f(x) = g(x) + h(Ax)
    # use g as the indicator fuction of R+
    # us h(y) = ||y-b||^2
    # We need tau*sigma<||A||_2^{-2}, use Frobenius norm as upper bound for 2 norm
    Aop = scipy.sparse.linalg.aslinearoperator(A)
    Anorm2 = 0
    for i in range(0, A.shape[1]):
        x = np.zeros(A.shape[1])
        x[i] = 1
        y = Aop.matvec(x)
        Anorm2 = Anorm2 + np.dot(y,y)
        
    #Anorm2 = np.dot(A.ravel(),A.ravel())
    sigma = 0.9/np.sqrt(Anorm2)
    tau = 0.9/np.sqrt(Anorm2)
    theta = 1.0

    tausig = tau*sigma
    half_sig = sigma/2
    s1 = 1.0/(1.0+half_sig)
    s2 = half_sig*s1

    x = np.zeros(A.shape[1])
    y = np.zeros(A.shape[0])
    u = np.zeros(A.shape[0])
    uold = np.zeros(A.shape[0])
    ubar = np.zeros(A.shape[0])

    tol = 1e-5
    max_iter = 1000
    diff = np.inf
    iter = 0
    while ( diff>tol and iter<max_iter ):
        iter += 1
        #print x,y,u,ubar
        xold = x
        x = np.clip( x - tausig*Aop.rmatvec(ubar), 0.0, np.inf )
        Ax = Aop.matvec(x)
        y = s1*b + s2*(u+Ax)
        uold = u
        u = uold + Ax - y
        ubarold = ubar
        ubar = u + theta*(u-uold)

        diff = np.linalg.norm(ubar-ubarold)
        
        #print 'iter =', iter, np.linalg.norm(Aop.matvec(x)-b)
        
    return x
    

def pdhgmp2_nnlsqr(A,b):
    # Algorithm 8 in
    # First Order Algorithms in Variational Image Processing
    # minimise f(x) = g(x) + h(Mx)
    # use M = Id
    # use g as the indicator fuction of R+
    # us h(x) = ||Ax-b||^2
    # We need tau*sigma<||M||_2^{-2}, M is the identity...
    sigma = 0.9
    tau = 0.9
    theta = 1.0

    tausig = tau*sigma
    sqrt_half_sig = np.sqrt(sigma/2)

    x = np.zeros(A.shape[1])
    y = np.zeros(A.shape[1])
    u = np.zeros(A.shape[1])
    uold = np.zeros(A.shape[1])
    ubar = np.zeros(A.shape[1])

    Adamped = scipy.sparse.linalg.LinearOperator(
        A.shape+np.array([A.shape[1], 0]),
        matvec = lambda y: np.concatenate([ A.dot(y), sqrt_half_sig*y ]),
        rmatvec = lambda y: y[0:A.shape[0]].dot(A) + sqrt_half_sig*y[A.shape[0]:],
        )

    tol = 1e-5
    max_iter = 500
    diff = np.inf
    iter = 0
    while ( diff>tol and iter<max_iter ):
        iter += 1
        #print x,y,u,ubar
        x = np.clip( x - tausig*ubar, 0, np.inf )
        y = lsmr_with_init( Adamped, np.concatenate([ b, sqrt_half_sig*(u+x) ]), x )
        uold = u
        u = uold + x - y
        ubar = u + theta*(u-uold)

        diff = np.linalg.norm(u-uold)
        
        #print 'iter =', iter, np.linalg.norm(A.dot(x)-b)
        
    return x
    

def dogbox_nnlsqr(A,b):
    # A Rectangular Trust Region Dogleg Approach for
    # Unconstrained and Bound Constrained Nonlinear Optimization 
    Aop = scipy.sparse.linalg.aslinearoperator(A)

    def isfeasible(xx, xlb, xub):
        return np.all(np.concatenate([xx>=xlb, xx<=xub]))

    x = np.zeros(A.shape[1])
    #passive_set = np.ones(A.shape[1], dtype=bool)
    delta = np.inf

    tol = 1e-5
    max_iter = 5
    iter = 0
    while (iter<max_iter ):
        iter += 1

        lb = np.fmax(-x,-delta)
        ub = delta*np.ones(A.shape[1])

        b_centered = b-Aop.matvec(x)
        mg = Aop.rmatvec(b_centered)

        #print 'iter', iter
        #print 'lb', lb
        #print 'ub', ub
        #print 'mg', mg
        #print 'x', x
        
        passive_set = ~( ( (x<=lb) & (mg<0.0) ) | ( (x>=ub) & (mg>0.0) ) )
        #print 'passive_set',passive_set

        def xreduce(xx):
            return xx[passive_set]

        def xexpand(xred):
            xe = np.zeros(A.shape[1])
            xe[passive_set] = xred
            return xe

        Ared = scipy.sparse.linalg.LinearOperator(
            np.array([A.shape[0], np.count_nonzero(passive_set)]),
            matvec = lambda y: Aop.matvec(xexpand(y)),
            rmatvec = lambda y: xreduce(Aop.rmatvec(y)),
            )

        xrold = xreduce(x)
        br_centered = b-Ared.matvec(xrold)
    
        xr_pack = scipy.sparse.linalg.lsmr(Ared,br_centered)
        xr_newton = xr_pack[0];

        lbr = xreduce(lb)
        ubr = xreduce(ub)

        if isfeasible(xr_newton, lbr, ubr):
            sr = xr_newton
        else:
            mgr = Ared.rmatvec(b)
            gTgr = np.dot(mgr,mgr)
            mAgr = Ared.matvec(mgr)
            gTATAgr = np.dot(mAgr,mAgr)

            xr_cauchy = (gTgr/gTATAgr)*mgr

            if isfeasible(xr_cauchy, lbr, ubr):
                NnC = xr_newton-xr_cauchy;

                idx = (NnC>0.0)
                if np.all(idx):
                    alpha = np.min( (ubr[idx] - xr_cauchy[idx]) / NnC[idx] )
                elif np.all(~idx):
                    alpha = np.min( (lbr[~idx] - xr_cauchy[~idx]) / NnC[~idx] )
                else:
                    alphau = np.min( (ubr[idx] - xr_cauchy[idx]) / NnC[idx] )
                    alphal = np.min( (lbr[~idx] - xr_cauchy[~idx]) / NnC[~idx] )
                    alpha = np.fmin(alphau,alphal)
                #print 'alpha', alphau, alphal

                sr =  xr_cauchy + alpha*NnC
            else:
                idx = (xr_cauchy>0.0)
                if np.all(idx):
                    beta = np.min( ubr[idx] / xr_cauchy[idx] )
                elif np.all(~idx):
                    beta =  np.min( lbr[~idx] / xr_cauchy[~idx] )
                else:
                    betau = np.min( ubr[idx] / xr_cauchy[idx] )
                    betal = np.min( lbr[~idx] / xr_cauchy[~idx] )
                    beta = np.fmin(betau,betal)
                #print 'beta', betau, betal

                PC = beta*xr_cauchy
                NnPC = xr_newton-PC;

                idx = (NnPC>0.0)
                if np.all(idx):
                    alpha = np.min( (ubr[idx] - PC[idx]) / NnPC[idx] )
                elif np.all(~idx):
                    alpha = np.min( lbr[~idx] - PC[~idx] / NnPC[~idx] )
                else:
                    alphau = np.min( (ubr[idx] - PC[idx]) / NnPC[idx] )
                    alphal = np.min( lbr[~idx] - PC[~idx] / NnPC[~idx] )
                    alpha = np.fmin(alphau,alphal)
                #print 'alpha', alphau, alphal

                sr =  PC + alpha*NnPC

        if ( np.dot(sr,sr)<tol ):
            break
                
        x = x + xexpand(sr)
        
        #print 'iter =', iter, np.linalg.norm(A.dot(x)-b)

    return x
    

def bpp_nnlsqr(A,b):
    # similar to TSNNLS
    Aop = scipy.sparse.linalg.aslinearoperator(A)

    def isfeasible(xxf, yyf):
        return np.all(np.concatenate([xxf>=0, yyf>=0]))

    x = np.zeros(A.shape[1])
    y = -Aop.rmatvec(b)
    p = 3
    N = np.inf
    Fset = np.zeros(A.shape[1], dtype=bool)

    xF = x[Fset]
    yG = y[~Fset]

    iter = 0
    max_iter = 10
    while ( not isfeasible(xF, yG) and iter < max_iter ):
        iter = iter + 1

        n = np.count_nonzero(xF<0) + np.count_nonzero(yG<0)
        #print 'xf,yG = ', xF, yG, n

        if ( n<N ): # The number of infeasibles has decreased
            print 'The number of infeasibles has decreased'
            N = n
            p = 3
            # Exchange all infeasible variables between F and G
            #Fset[Fset][xF<0] = False
            #Fset[~Fset][yG<0] = True
            Fsetold = Fset
            if np.any(xF<0):
                Fset.flat[np.flatnonzero(Fsetold)[xF<0]] = False
            if np.any(yG<0):
                Fset.flat[np.flatnonzero(~Fsetold)[yG<0]] = True
        else:
            if ( p > 0 ):
                print 'p>0: ', p
                p = p-1
                # Exchange all infeasible variables between F and G
                Fsetold = Fset
                if np.any(xF<0): Fset.flat[np.flatnonzero(Fsetold)[xF<0]] = False
                if np.any(yG<0): Fset.flat[np.flatnonzero(~Fsetold)[yG<0]] = True
            else:
                print 'pick one'
                # Exchange only the infeasible variable with largest index
                tmp = np.concatenate([ np.ravel(np.where(Fset))[xF<0], np.ravel(np.where(~Fset))[yG<0] ])
                idx = np.max( tmp )
                Fset[idx] = (~Fset[idx])
        
        # update xF
        def xreduce(xx,set):
            return xx[set]

        def xexpand(xred,set):
            xe = np.zeros(A.shape[1])
            xe[set] = xred
            return xe

        AFop = scipy.sparse.linalg.LinearOperator(
            np.array([A.shape[0], np.count_nonzero(Fset)]),
            matvec = lambda y: Aop.matvec(xexpand(y,Fset)),
            rmatvec = lambda y: xreduce(Aop.rmatvec(y),Fset),
            )

        xF_pack = scipy.sparse.linalg.lsmr(AFop,b)
        xF = xF_pack[0]

        # update yG
        AGop = scipy.sparse.linalg.LinearOperator(
            np.array([A.shape[0], np.count_nonzero(~Fset)]),
            matvec = lambda y: Aop.matvec(xexpand(y,~Fset)),
            rmatvec = lambda y: xreduce(Aop.rmatvec(y),~Fset),
            )
            
        yG = AGop.rmatvec( AFop.matvec(xF) - b )

        # set small variables to 0
        xF[ np.abs(xF) < 1e-12 ] = 0.0
        yG[ np.abs(yG) < 1e-12 ] = 0.0

        x = np.zeros(A.shape[1])
        x[Fset] = xF
        
        #print 'iter =', iter, np.linalg.norm(A.dot(x)-b)

    return x
    

def admm_nnlsqr(A,b):
    # ADMM parameter
    # Optimal choice product of the maximum and minimum singular values
    # Heuristic choice: mean of singular values,
    # i.e. squared Frobenius norm over dimension
    Aop = scipy.sparse.linalg.aslinearoperator(A)
    Anorm2 = 0
    for i in range(0, A.shape[1]):
        x = np.zeros(A.shape[1])
        x[i] = 1
        y = Aop.matvec(x)
        
    #rho = np.dot(A.ravel(),A.ravel()) / A.shape[1]
    rho = Anorm2 / A.shape[1]
    sqrt_half_rho = np.sqrt(rho/2.0)
    #print 'sqrt_half_rho=',sqrt_half_rho
    #sqrt_half_rho = 100

    
    # initialisation
    zero_init = False
    if zero_init:
        x_pack = (np.zeros(A.shape[1]),0)
        z = np.zeros(A.shape[1])
        u = np.zeros(A.shape[1])
    else:
        # from x=z=u=0 the first iteration is a normally a projection of
        # the unconstrained damped least squares. Let's forget the damping and check
        # whether we need to constrain things
        x_pack = scipy.sparse.linalg.lsmr(Aop,b)

        if np.all(x_pack[0]>0):
            # no constraint needed
            return x_pack[0]
    
        z = np.clip(x_pack[0],0,np.inf)
        u = x_pack[0]-z

    # construct the damped least squares matrix
    # todo try and use directly the damped version of lsmr
    Adamped = scipy.sparse.linalg.LinearOperator(
        A.shape+np.array([A.shape[1], 0]),
        matvec = lambda y: np.concatenate([ Aop.matvec(y), sqrt_half_rho*y ]),
        rmatvec = lambda y: Aop.rmatvec(y[0:A.shape[0]]) + sqrt_half_rho*y[A.shape[0]:],
        )

    tol = 1e-5
    max_iter = 50
    diff = np.inf
    iter = 0
    while ( diff>tol and iter<max_iter ):
        iter += 1
        xold = x_pack[0]
        #x_pack = scipy.sparse.linalg.lsmr( Adamped, np.concatenate([ b, sqrt_half_rho*(z-u) ]) )
        x_pack = (lsmr_with_init( Adamped, np.concatenate([ b, sqrt_half_rho*(z-u) ]), xold ), 0)
        zold = z
        z = np.clip(x_pack[0]+u,0.0,np.inf)
        #diff = np.linalg.norm(z-zold)
        #diff = np.linalg.norm(x_pack[0]-xold)
        diff = np.linalg.norm(x_pack[0]-z)
        u += x_pack[0]-z
        #print 'iter', iter, ' -- diff', diff
        
        #print 'iter =', iter, np.linalg.norm(Aop.matvec(z)-b)
    
    return z

## A = np.array([[60, 90, 120],[30, 120, 90],[0, 12, 8],[-45, 8, 7]])
## b = np.array([67.5, -60, 8, 0.4])

## x_nnls, rnorm_nnls = scipy.optimize.nnls(A,b)
## print 'x   (nnls) =', x_nnls
## print 'norm       = ', np.linalg.norm(A.dot(x_nnls)-b)

## x_lstsq_pack = np.linalg.lstsq(A,b)
## x_lstsq = x_lstsq_pack[0]
## print 'x  (lstsq) =', x_lstsq
## print 'norm       = ', np.linalg.norm(A.dot(x_lstsq)-b)

## x_lsqlin_pack = scipy.optimize.lsq_linear(A,b,bounds=(0,np.inf),lsq_solver='lsmr')
## x_lsqlin = x_lsqlin_pack.x
## print 'x (lsqlin) =', x_lsqlin
## print 'norm       = ', np.linalg.norm(A.dot(x_lsqlin)-b), 'niter=', x_lsqlin_pack.nit

## x_lsmr_pack = scipy.sparse.linalg.lsmr(A,b)
## x_lsmr = x_lsmr_pack[0]
## print 'x   (lsmr) =', x_lsmr
## print 'norm       = ', np.linalg.norm(A.dot(x_lsmr)-b)

## x_lsqr_pack = scipy.sparse.linalg.lsqr(A,b)
## x_lsqr = x_lsqr_pack[0]
## print 'x   (lsqr) =', x_lsqr
## print 'norm       = ', np.linalg.norm(A.dot(x_lsqr)-b)

## x_nnlsqr = admm_nnlsqr(A,b)
## print 'x (nnlsqr) =', x_nnlsqr
## print 'norm       = ', np.linalg.norm(A.dot(x_nnlsqr)-b)


## x_pdhgmp = pdhgmp_nnlsqr(A,b)
## print 'x (pdhgmp) =', x_pdhgmp
## print 'norm       = ', np.linalg.norm(A.dot(x_pdhgmp)-b)


## x_pdhgmp2 = pdhgmp2_nnlsqr(A,b)
## print 'x (pdhgm2) =', x_pdhgmp2
## print 'norm       = ', np.linalg.norm(A.dot(x_pdhgmp2)-b)


## x_mfista = mfista_nnlsqr(A,b)
## print 'x (mfista) =', x_mfista
## print 'norm       = ', np.linalg.norm(A.dot(x_mfista)-b)


## x_dogbox = dogbox_nnlsqr(A,b)
## print 'x (dogbox) =', x_dogbox
## print 'norm       = ', np.linalg.norm(A.dot(x_dogbox)-b)


## x_fgr = fgr_nnlsqr(A,b)
## print 'x (fgr)    =', x_fgr
## print 'norm       = ', np.linalg.norm(A.dot(x_fgr)-b)


## x_bpp = bpp_nnlsqr(A,b)
## print 'x (bpp)    =', x_bpp
## print 'norm       = ', np.linalg.norm(A.dot(x_bpp)-b)



lena = scipy.ndimage.imread('../data/test/2D_Lena_256.png','F')
lena = lena[100:150,100:150]
im_size = lena.shape
num_pix = lena.size
print 'num_pix = ', num_pix, im_size

sigma = 2
BlurOp = scipy.sparse.linalg.LinearOperator(
        [num_pix, num_pix],
        matvec = lambda y: scipy.ndimage.filters.gaussian_filter( y.reshape(im_size), sigma ),
        rmatvec = lambda y: scipy.ndimage.filters.gaussian_filter( y.reshape(im_size), sigma ),
        )

lena_blurred = BlurOp.matvec(lena.ravel())

start = time.time()
lres_pack = scipy.sparse.linalg.lsmr(BlurOp,lena_blurred)
end = time.time()
lres = lres_pack[0]
lres_lsmr = lres
print ' '
print 'lsmr - ', end-start, 'sec'
print 'min = ', np.min(lres)
print 'max = ', np.max(lres)
print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
print 'err2 = ', np.linalg.norm(lres-lena.ravel())


start = time.time()
lres = mfista_nnlsqr(BlurOp,lena_blurred)
end = time.time()
lres_mfista = lres
print ' '
print 'mfista - ', end-start, 'sec'
print 'min = ', np.min(lres)
print 'max = ', np.max(lres)
print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
print 'err2 = ', np.linalg.norm(lres-lena.ravel())


start = time.time()
lres = dogbox_nnlsqr(BlurOp,lena_blurred)
end = time.time()
lres_dogbox = lres
print ' '
print 'dogbox - ', end-start, 'sec'
print 'min = ', np.min(lres)
print 'max = ', np.max(lres)
print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
print 'err2 = ', np.linalg.norm(lres-lena.ravel())


start = time.time()
lres = bpp_nnlsqr(BlurOp,lena_blurred)
end = time.time()
lres_bpp = lres
print ' '
print 'bpp - ', end-start, 'sec'
print 'min = ', np.min(lres)
print 'max = ', np.max(lres)
print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
print 'err2 = ', np.linalg.norm(lres-lena.ravel())


start = time.time()
lres = fgr_nnlsqr(BlurOp,lena_blurred)
end = time.time()
lres_fgr = lres
print ' '
print 'fgr - ', end-start, 'sec'
print 'min = ', np.min(lres)
print 'max = ', np.max(lres)
print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
print 'err2 = ', np.linalg.norm(lres-lena.ravel())


start = time.time()
lres = admm_nnlsqr(BlurOp,lena_blurred)
end = time.time()
lres_admm = lres
print ' '
print 'admm - ', end-start, 'sec'
print 'min = ', np.min(lres)
print 'max = ', np.max(lres)
print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
print 'err2 = ', np.linalg.norm(lres-lena.ravel())


start = time.time()
lres = pdhgmp_nnlsqr(BlurOp,lena_blurred)
end = time.time()
lres_pdhgmp = lres
print ' '
print 'pdhgmp - ', end-start, 'sec'
print 'min = ', np.min(lres)
print 'max = ', np.max(lres)
print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
print 'err2 = ', np.linalg.norm(lres-lena.ravel())


## start = time.time()
## lres_pack = scipy.optimize.lsq_linear(BlurOp,lena_blurred,bounds=(0,np.inf),lsq_solver='lsmr')
## end = time.time()
## lres = lres_pack.x
## lres_lsql = lres
## print ' '
## print 'lsql - ', end-start, 'sec'
## print 'min = ', np.min(lres)
## print 'max = ', np.max(lres)
## print 'err = ', np.linalg.norm(BlurOp.matvec(lres)-lena_blurred)
## print 'err2 = ', np.linalg.norm(lres-lena.ravel())


plt.figure(1)
ax = plt.subplot(3,3,1)
ax.set_title('Original')
plt.imshow(lena,cmap='gray')

ax = plt.subplot(3,3,2)
ax.set_title('Blurred')
plt.imshow(lena_blurred.reshape(im_size),cmap='gray')

ax = plt.subplot(3,3,3)
ax.set_title('LSMR')
plt.imshow(lres_lsmr.reshape(im_size),cmap='gray')

#plt.subplot(2,3,4)
#plt.imshow(lres_lsql.reshape(im_size),cmap='gray')

ax = plt.subplot(3,3,4)
ax.set_title('BPP')
plt.imshow(lres_bpp.reshape(im_size),cmap='gray')

ax = plt.subplot(3,3,5)
ax.set_title('MFISTA (LSMR init)')
plt.imshow(lres_mfista.reshape(im_size),cmap='gray')

ax = plt.subplot(3,3,6)
ax.set_title('DOGBOX')
plt.imshow(lres_dogbox.reshape(im_size),cmap='gray')

ax = plt.subplot(3,3,7)
ax.set_title('FGR (zero init)')
plt.imshow(lres_fgr.reshape(im_size),cmap='gray')

ax = plt.subplot(3,3,8)
ax.set_title('ADMM')
plt.imshow(lres_admm.reshape(im_size),cmap='gray')

ax = plt.subplot(3,3,9)
ax.set_title('PDHGMP')
plt.imshow(lres_pdhgmp.reshape(im_size),cmap='gray')

plt.show()
