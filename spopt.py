"""
spopt is a solver for Optimization on the Symplectic Stiefel manifold:
min f(X), s.t., X'*J_{2n}*X = J_{2k},

where $X \in R^{2n,2k} and J_{2k} = [zeros(k,k),eye(k);-eye(k),zeros(k,k)]$

------- main iterative update: X(tau) -------
P = I - J*X*inv(X'*X)*X'*J' + 0.5*pm*X*X';
(or)
P = (I-X*J*X'*J')(I-X*J*X'*J')' + 0.5*pm*X*X';
U = [-P*G, X*J];
V = [X*J, -P*G];
X(tau) = X + tau*U*inv(I+0.5*tau*V'*J'*U)*V'*J*X
---------------------------------------------

Input:
    X --- 2n by 2k matrix such that X'*J_{2n}*X = J_{2k}
    fun --- objective function and its gradient:
             [F, G] = fun(X,  data1, data2)
             F, G are the objective function value and gradient, respectively
             data1, data2 are additional data, and can be more
             Calling syntax:
               [X, out]= OptSymplecticGBB(X0, @fun, opt, data1, data2);

   opt --- option structure with fields:
             record      0: no print out
             mxitr       max number of iterations
             xtol        stop control for ||X_k - X_{k-1}||
             gtol        stop control for the projected gradient
             ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
             retr        retraction map: 1 for Cayley, o.w. quasi-geodesic

Output:
    X --- solution
    Out --- output information
             feaX(final feasibility)     --  feaXs(iterative history)
             nrmG(final gradient norm)   --  kkts(iterative history)
             fval(final function value)  --  fvals(iterative history)
             itr(final iteration number) --  times(time line for iterations)
-------------------------------------
Reference:
Bin Gao, Nguyen Thanh Son, P.-A. Absil, Tatjana Stykel
1. Riemannian optimization on the symplectic Stiefel manifold (https://arxiv.org/abs/2006.15226)
2. Riemannian gradient method on the symplectic Stiefel manifold based on the Euclidean metric
Author: Bin Gao (https://www.gaobin.cc)
Version 0.1 ... 2019/11
Version 0.2 ... 2020/03: add Eucliean metric
Version 1.0 ... 2020/06: Release at github: https://github.com/opt-gaobin/spopt
"""
import numpy as np
import numpy.linalg as lin
import argparse
import time
from control.matlab import *
from scipy.linalg import expm

def feval(funcName, *args):
    """
    This function is similar to "feval" in Matlab.
    Example: feval('cos', pi) = -1.
    """
    # return eval(funcName)(*args)
    return lin.norm(args[0] - args[1], 'fro') ** 2, 2 * (args[0] - args[1])


def spopt(X=None, fun=None, opt=None, *args, **kwargs):
    # for output
    parser = argparse.ArgumentParser(description='For output')
    out = parser.parse_args()

    # Problem size
    if X.any():
        nn, kk = X.shape
        n = round(nn / 2)
        k = round(kk / 2)
    else:
        raise Exception('Input X is an empty matrix')

    # Copy parameters
    gtol = opt.gtol
    rhols = opt.rhols
    eta = opt.eta
    gamma = opt.gamma
    tau = opt.tau
    maxtau = opt.maxtau
    mintau = opt.mintau
    record = opt.record
    nt = opt.nt
    retr = opt.retr
    pm = opt.pm
    pg = opt.pg
    metric = opt.metric

    # Save metric and solver
    if metric != 1:
        retr = 2
        metricname = 'Euclidean'  # Cayley retraction cannot be applied to Euclidean metric
    else:
        metricname = 'Canonical-like'

    if retr == 1:
        retrname = 'Cayley'
    else:
        retrname = 'quasi-geodesic'

    # ------------------------------------------------------------------------
    # Initialization
    J2k = np.concatenate([np.concatenate([np.zeros((k, k)), np.eye(k)], axis=1),
                          np.concatenate([-np.eye(k), np.zeros((k, k))], axis=1)], axis=0)

    # Evaluate function and gradient info.
    F, G = feval(fun, X, args[0])  # this is a problem, figure it out
    out.nfe = 1

    # Preparations for the first update
    XJ = np.concatenate([-X[:, k:], X[:, :k]], axis=1)
    JX = np.concatenate([X[n:, :], -X[:n, :]], axis=0)
    if metric == 1:
        GX = 0.5 * pm * np.matmul(G.transpose(), X)
        if k < n:
            if pg == 1:
                XX = np.matmul(X.transpose(), X)
                invXXXJG = lin.solve(np.matmul(XX.transpose(), XX), np.matmul(XX.transpose(), np.matmul(JX.transpose(), G)))
                PG = G - np.matmul(JX, invXXXJG) + np.matmul(X, GX.transpose())
            else:
                XJG = np.matmul(XJ.transpose(), G)
                JXXJG = np.matmul(JX, XJG)
                PG = G - JXXJG - np.matmul(XJ, (np.matmul(JX.transpose(), G))) + np.matmul(XJ, (
                    np.matmul(JX.transpose(), JXXJG))) + np.matmul(X, GX.transpose())
        else:
            PG = np.matmul(X, GX.transpose())
    else:
        XX = np.matmul(X.transpose(), X)
        JXG = np.matmul(JX.transpose(), G)
        skewXJG = JXG - JXG.transpose()

    if retr == 1:
        invH = True
        eye2n = np.eye(2 * n)
        if k < n / 2:
            invH = False
            eye2k = np.eye(2 * k)
            eye4k = np.eye(4 * k)
        if invH:
            PGXJ = np.matmul(- PG, XJ.transpose())
            H = PGXJ + PGXJ.transpose()
            HJ = np.concatenate([-H[:, n:], H[:, :n]], axis=1)
            RJX = np.matmul(H, JX)
        else:
            U = np.concatenate([-PG, XJ], axis=1)
            PGJPG = np.matmul(np.concatenate([PG[n:, :], -PG[:n, :]], axis=0).transpose(), PG)
            VJU = np.concatenate(
                [np.concatenate([GX.transpose(), J2k.transpose()], axis=1), np.concatenate([PGJPG, - GX], axis=1)],
                axis=0)
            VJX = np.concatenate([eye2k, np.concatenate([GX[:, k:], -GX[:, :k]], axis=1)], axis=0)
            # Direct way to get VJU and VJX
            # U =  [-PG, XJ]; V = [XJ, -PG];	VJU = V'*[-U(n+1:end,:); U(1:n,:)];
            # VJX = V'*JX;
    else:
        eye2k = np.eye(2 * k)
        if metric == 1:
            W = np.concatenate([-GX[:, k:], GX[:, :k]], axis=1)
            W = W + W.transpose()
            # Direct way to get W
            # XJG = XJ'*G; W = 0.5*pm*(XJG+XJG');
        else:
            Omega = lyap(XX, - skewXJG)  # fix this
            W = - JXG + np.matmul(XX, Omega)
            W = np.matmul(-0.5, (W + W.transpose()))

    # Compute initial error
    XFeasi = lin.norm(np.matmul(X.transpose(), JX) - J2k, 'fro')
    if metric == 1:
        dtX = np.matmul(PG, (np.matmul(XJ.transpose(), JX))) + np.matmul(XJ, (np.matmul(PG.transpose(), JX)))
    else:
        dtX = G - np.matmul(JX, Omega)

    nrmG = lin.norm(dtX, 'fro')
    # nrmG0 = nrmG  # Initial gradient norm

    # Save history
    out.fvals = []
    out.fvals.append(F)
    out.kkts = []
    out.kkts.append(nrmG)
    out.feaXs = []
    out.feaXs.append(XFeasi)
    out.times = []
    out.times.append(0)
    t = time.time()

    # Line-search parameter
    Q = 1
    Cval = F

    # Print info.
    if opt.record == 1:
        fid = 1
        print('------------------------------------------------------------------------')
        print('Solver setting... ({} metric, {} retraction)'.format(metricname, retrname))
        print('----------- Riemannian Gradient Method with Line search ----------------- ')
        print('{:<7s} {:<7} {:<7s} {:<7s} {:<7s} {:<7s} {:<7s}'.format('Iter', 'tau', 'F(X)', 'nrmG', 'XDiff', 'FDiff', 'XFeasi'))
        print('{:<7}  {:<7}  {:<7.3e}  {:<7.3e}  {:<7}  {:<7}  {:<7.3e}  {:<7}'.format(0, '.', F, nrmG, '.', '.', XFeasi, '.'))

    # ------------------------------------------------------------------------
    # Main iteration
    for itr in range(opt.mxitr):
        XP = X
        FP = F
        dtXP = dtX  # GP = G

        # -------- Scale step size by non-monotone line-search --------
        nls = 1
        deriv = rhols * abs(iprod(G, dtXP))  # Riemannian metric ||dtXP||^2_X
        while 1:
            # ----- Retraction step -----
            if retr == 1:
                # Cayley
                if invH:
                    X = lin.solve(np.matmul((eye2n - (tau * HJ)).transpose(), eye2n - (tau * HJ)), np.matmul((eye2n - (tau * HJ)).transpose(), XP + (tau * RJX)))
                else:
                    aa = lin.solve(np.matmul((eye4k + (0.5 * tau * VJU)).transpose(), eye4k + (0.5 * tau * VJU)), np.matmul((eye4k + (0.5 * tau * VJU)).transpose(), VJX))
                    X = XP + np.matmul(U, (tau * aa))
            else:
                # Quasi-geodesic
                U = np.concatenate([XP, -tau * dtX], axis=1)
                JWt = tau * np.concatenate([W[k:, :], -W[:k, :]], axis=0)
                if nls == 1:
                    JZJZ = np.matmul(np.concatenate([-dtX[:, k:], dtX[:, :k]], axis=1).transpose(),
                                     np.concatenate([dtX[n:, :], -dtX[:n, :]], axis=0))
                H = np.concatenate([np.concatenate([-JWt, (-(tau ** 2) * JZJZ)], axis=1),
                                    np.concatenate([eye2k, -JWt], axis=1)], axis=0)
                ExpH = expm(H)
                X = np.matmul(U, (np.matmul(ExpH[:, :2 * k], expm(JWt))))

                # Direct computation based on quasi-geodesic curve
                # U = [X,-dtX]; JW = [W(k+1:end,:); -W(1:k,:)];
                # if nls == 1; JZJZ = [-dtX(:,k+1:end) dtX(:,1:k)]'*[dtX(n+1:end,:); -dtX(1:n,:)]; end;
                # H = [-JW -JZJZ; eye2k -JW]; ExpH = lin.expm(tau*H);
                # X = U*(ExpH(:,1:2*k)*lin.expm(tau*JW));

            # ----- Evaluate function -----
            F, G = feval(fun, X, args[0])
            out.nfe = out.nfe + 1
            # ----- line search --------
            if F <= Cval - (tau* deriv) or nls >= nt:
                break
            tau = eta * tau
            nls = nls + 1

        # -------------------- Prepare retraction --------------------
        XJ = np.concatenate([- X[:, k:], X[:, :k]], axis=1)
        JX = np.concatenate([X[n:, :], -X[:n, :]], axis=0)
        if metric == 1:
            GX = 0.5 * pm * np.matmul(G.transpose(), X)
            if k < n:
                if pg == 1:
                    XX = np.matmul(X.transpose(), X)
                    invXXXJG = lin.solve(np.matmul(XX.transpose(), XX), np.matmul(XX.transpose(), np.matmul(JX.transpose(), G)))
                    PG = G - np.matmul(JX, invXXXJG) + np.matmul(X, GX.transpose())
                else:
                    XJG = np.matmul(XJ.transpose(), G)
                    JXXJG = np.matmul(JX, XJG)
                    PG = G - JXXJG - np.matmul(XJ, (np.matmul(JX.transpose(), G))) + np.matmul(XJ, (
                        np.matmul(JX.transpose(), JXXJG))) + np.matmul(X, GX.transpose())
            else:
                PG = np.matmul(X, GX.transpose())
        else:
            XX = np.matmul(X.transpose(), X)
            JXG = np.matmul(JX.transpose(), G)
            skewXJG = JXG - JXG.transpose()

        if retr == 1:
            if invH:
                PGXJ = np.matmul(-PG, XJ.transpose())
                H = 0.5 * (PGXJ + PGXJ.transpose())
                HJ = np.concatenate([-H[:, n:], H[:, :n]], axis=1)
                RJX = np.matmul(H, JX)
            else:
                U = np.concatenate([- PG, XJ], axis=1)
                PGJPG = np.matmul(np.concatenate([PG[n:, :], -PG[:n, :]], axis=0).transpose(), PG)
                VJU = np.concatenate([np.concatenate([GX.transpose(), J2k.transpose()], axis=1), np.concatenate([PGJPG, -GX], axis=1)], axis=0)
                VJX = np.concatenate([eye2k, np.concatenate([GX[:, k:], -GX[:, :k]], axis=1)], axis=0)
                # Direct way to get VJU and VJX
                # U =  [-PG, XJ]; V = [XJ, -PG];	VJU = V'*[-U(n+1:end,:); U(1:n,:)];
                # VJX = V'*JX;
        else:
            if metric == 1:
                W = np.concatenate([-GX[:, k:], GX[:, :k]], axis=1)
                W = W + W.transpose()
                # direct way to get W
                # XJG = XJ'*G; W = 0.5*pm*(XJG+XJG');
            else:
                Omega = lyap(XX, - skewXJG)
                W = - JXG + np.matmul(XX, Omega)
                W = np.matmul(- 0.5, (W + W.transpose()))

        # ---------------------- Compute error ----------------------
        S = X - XP
        XDiff = lin.norm(S, 'fro') / np.sqrt(n)
        tauk = tau
        FDiff = abs(FP - F) / (abs(FP) + 1)
        XFeasi = lin.norm(np.matmul(X.transpose(), JX) - J2k, 'fro')
        if metric == 1:
            dtX = np.matmul(PG, (np.matmul(XJ.transpose(), JX))) + np.matmul(XJ, (np.matmul(PG.transpose(), JX)))
        else:
            dtX = G - np.matmul(JX, Omega)
        nrmG = lin.norm(dtX, 'fro')

        out.fvals.append(F)
        out.kkts.append(nrmG)
        out.feaXs.append(XFeasi)
        out.times.append(time.time() - t)

        # print history
        if (record >= 1 and np.mod(itr+1, 15)) == 0:
            print('{:<7}  {:<7.3e} {:<7.3e}  {:<7.3e}  {:<7.3e}  {:<7.3e}  {:<7.3e}  {:<7.3e}'
                  .format(itr+1, tauk, F, nrmG, XDiff, FDiff, XFeasi, nls))

        # -------------------- Update step size ---------------------
        if 1 == opt.stepsize:  # Alternating BB
            # Y = G - GP;     SY = abs(iprod(S,Y));
            Y = dtX - dtXP
            SY = abs(iprod(S, Y))
            if np.mod(itr, 2) == 0:
                tau = (lin.norm(S, 'fro') ** 2) / SY
            else:
                tau = SY / (lin.norm(Y, 'fro') ** 2)
        elif 2 == opt.stepsize:  # BB1
            Y = dtX - dtXP
            SY = abs(iprod(S, Y))
            tau = (lin.norm(S, 'fro') ** 2) / SY
        elif 3 == opt.stepsize:  # BB2
            Y = dtX - dtXP
            SY = abs(iprod(S, Y))
            tau = SY / (lin.norm(Y, 'fro') ** 2)
        elif 4 == opt.stepsize:  # Noceldal & Wright, p59(3.60)
            tau = abs(2 * (FP - F) / iprod(G, dtX))
            # tau = abs(2*(FP - F)/nrmG^2); # Euclidean
        tau = max(min(tau, maxtau), mintau)

        # --------------------- stop criteria -----------------------------------
        #     crit(itr,:) = [nrmG, XDiff, FDiff];
        #     mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
        if nrmG < gtol:
            # if nrmG < gtol*nrmG0
            # if (XDiff < xtol && nrmG < gtol ) || FDiff < ftol
            # if (XDiff < xtol || nrmG < gtol ) || FDiff < ftol
            # if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol
            # if ( XDiff < xtol || FDiff < ftol ) || nrmG < gtol
            # if any(mcrit < [gtol, xtol, ftol])
            # if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
            out.msg = 'converge'
            break
        # --------------------- nonmonotone update --------------------
        Qp = Q
        Q = (gamma * Qp) + 1
        Cval = ((gamma * Qp * Cval) + F) / Q

    # ------------------------------------------------------------------------
    # Output
    if itr >= opt.mxitr:
        out.msg = 'exceed max iteration'

    if record >= 1:
        print('{} at...'.format(out.msg))
        print('{:<7}  {:<7.3e}  {:<7.3e}  {:<7.3e}  {:<7.3e}  {:<7.3e}  {:<7.3e}  {:<7.3e}'
              .format(itr+1, tauk, F, nrmG, XDiff, FDiff, XFeasi, nls))
        print('------------------------------------------------------------------------')

    out.feaX = XFeasi
    out.nrmG = nrmG
    out.fval = F
    out.itr = itr
    return X, out


# Nest-function: inner product
def iprod(x=None, y=None, *args, **kwargs):
    # a = real(sum(sum(conj(x).*y)));
    return np.real(sum(sum(np.multiply(x, y))))


if __name__ == '__main__':
    pass
