# Generated with SMOP  0.41-beta
# from libsmop import *
# demo_spopt.m
from spopt import *

import matplotlib.pyplot as plt
from scipy.linalg import expm


# -------------------------------------------------------------------------
# This demo shows how to call spopt to solve
#       min  f(X), s.t.  X'*J2n*X=J2k.
# where J2k = [zeros(k,k),eye(k);-eye(k),zeros(k,k)].
# -------------------------------------
# objective:    nearest symplectic matrix problem, f(X):= norm(X-A,'fro')^2
# solver:       Cayley and quasi-geodesic retraction (Canonical-like metric)
# output:       function information, iterative figures
# -------------------------------------
# Author: Bin Gao (https://www.gaobin.cc)
#   Version 1.0 ... 2020/06
# --------------------------------------------------------------------------
# objective function

def fun(X=None, A=None, *args, **kwargs):
    F = np.linalg.norm(X - A, 'fro') ** 2
    G = 2 * (X - A)
    return F, G


if __name__ == '__main__':
    pass

    # --- Problem generation ---
    n = 200
    k = 40
    # --- scenario 1:
    # A is randomly generated as a perturbation (in the normal space) of
    # a symplectic matrix, which means that the problem has the closed-form
    # solution, and achieves the optimal function value f_star = 0.
    # J2n = [zeros(n) eye(n);-eye(n) zeros(n)]; J2k = [zeros(k) eye(k);-eye(k) zeros(k)];
    # WA = randn(2*n,2*n); WA = WA'*WA+0.1*eye(2*n); EA = expm([WA(n+1:end,:); -WA(1:n,:)]);
    # A = [EA(:,1:k) EA(:,n+1:n+k)];
    # s = 1e-8; K = rand(2*k,2*k); K = K - K'; B = A; A = B + s*J2n*(B*K);

    # --- scenario 2:
    # A is totally randomly generated
    A = np.random.randn(2 * n, 2 * k)
    A = A / np.linalg.norm(A, 2)

    parser = argparse.ArgumentParser(description='Symplectic Stiefel Optimizer')

    parser.add_argument("-xtol", type=int, default=1e-6, help="stop control for ||X_k - X_{k-1}||")
    parser.add_argument("-gtol", type=int, default=1e-6, help="stop control for the projected gradient")
    parser.add_argument("-ftol", type=int, default=1e-12, help="stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)")
    parser.add_argument("-mxitr", type=int, default=1e3, help="max number of iterations")
    parser.add_argument("-record", type=int, default=0, help="0: no print out")

    # Parameters for control the linear approximation in line search,
    parser.add_argument("-tau", type=int, default=1e-3, help="initial step-size")
    parser.add_argument("-maxtau", type=int, default=1e5, help="maximal step-size")
    parser.add_argument("-mintau", type=int, default=1e-5, help="minimal stepsize")
    parser.add_argument("-rhols", type=int, default=1e-4, help="linear search condition")
    parser.add_argument("-eta", type=int, default=0.1, help="back tracking parameter")
    parser.add_argument("-gamma", type=int, default=0.85, help="non-monotone parameter")
    parser.add_argument("-nt", type=int, default=5, help="max number of linear search steps")
    parser.add_argument("-stepsize", type=int, default=1, help="different strategies for stepsize")
    parser.add_argument("-retr", type=int, default=1, help="parameter for Riemannian optimization")
    parser.add_argument("-pg", type=int, default=1, help="different choices of Riemannian gradient")
    parser.add_argument("-metric", type=int, default=1, help="1: canonical like, o.w.: Euclidean")

    opt = parser.parse_args()
    if opt.pg == 1:
        opt.pm = 0.5  # parameter of canonical-like metric
    else:
        opt.pm = 1

    # --- parameters ---
    opt.record = 1
    opt.mxitr = 1000
    opt.gtol = 1e-06

    # --- generate initial guess ---
    # type 1: "identity"
    # X0 = zeros(2*n,2*k); X0(1:k,1:k) = eye(k); X0(n+1:n+k,k+1:end) = eye(k);
    # type 2: random
    W = np.random.randn(2 * k, 2 * k)

    W = np.matmul(W.transpose(), W) + 0.1 * np.eye((2 * k))

    E = expm(np.concatenate([W[k:, :], -W[:k, :]], axis=0))
    X0 = np.concatenate([E[:k, :],
                         np.zeros((n - k, 2 * k)),
                         E[k:, :],
                         np.zeros((n - k, 2 * k))], axis=0)

    # call solver
    # --- Cayley retraction ---
    st1 = time.time()
    __, out1 = spopt(X0, fun, opt, A)
    ste1 = time.time() - st1

    # --- Quasi-geodesic ---
    opt.retr = 2
    st2 = time.time()
    __, out2 = spopt(X0, fun, opt, A)
    ste2 = time.time() - st2

    print('spopt-cay: obj: {:.3e}, itr: {}, nrmG: {:.3e}, nfe: {}, time: {:.3f}, |X^T J X - J|: {:.3e}'
          .format(out1.fval, out1.itr, out1.nrmG, out1.nfe, ste1, out1.feaX))
    print('spopt-geo: obj: {:.3e}, itr: {}, nrmG: {:.3e}, nfe: {}, time: {:.3f}, |X^T J X - J|: {:.3e}'
          .format(out2.fval, out2.itr, out2.nrmG, out2.nfe, ste2, out2.feaX))

    # figure
    # --- function value ---
    f_fval, ax = plt.subplots()
    ax.plot(out1.times, out1.fvals, 'r-', linewidth=1.5, label='Sp-Cayley')
    ax.plot(out2.times, out2.fvals, 'b--', linewidth=1.5, label='Quasi-geodesic')
    plt.yscale('log', basey=10)
    plt.xlabel('time(s)')
    plt.ylabel('function value')
    plt.title(r'$Size: {} \times {} $'.format(2 * n, 2 * k))
    plt.legend()
    plt.show()

    # --- gradient ---
    f_kkt, ax = plt.subplots()
    ax.plot(out1.times, out1.kkts, 'r-o', linewidth=1.5, label='Sp-Cayley')
    ax.plot(out2.times, out2.kkts, 'b--+', linewidth=1.5, label='Quasi-geodesic')
    plt.yscale('log', basey=10)
    plt.xlabel('time (s)')
    plt.ylabel(r'$||\nabla f||$')
    plt.title(r'$Size: {} \times {} $'.format(2 * n, 2 * k))
    plt.legend()
    plt.show()
