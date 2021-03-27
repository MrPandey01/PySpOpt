Python implementation of SpOpt package (originally in Matlab): Reimannian Optimization on Symplectic Stiefel Manifold 

# PySpOpt
A Python solver for Riemannian Optimization on the Symplectic Stiefel manifold. This was originally written in Matlab by [1].

## Problems
Solves the following optimization problem,
```math
$ \min f(X), s.t.   X'J2n X = J2p, $
```
  
where X is a 2n-by-2p matrix, J2n = [0 I_n; -I_n 0], and I_n is the n-by-n identity matrix.

## Applications
1. The nearest symplectic matrix problem:

> min ||X-A||^2_F, s.t.  X' J2n X = J2p.

2. The extrinsic mean problem:
  
> min 1/N sum_{i=1}^{i=N} ||X - A_i||^2_F, s.t.  X' J2n X = J2p.
  
3. Minimization of the Brockett cost function:

> min trace(X'AXN-2BX'), s.t.  X' J2n X = J2p.
  
4. Symplectic eigenvalue problem:

> min trace(X'AX), s.t.  X' J2n X = J2p.
  
5. Symplectic model order reduction:

> min ||M-XX^\dag M||, s.t.  X' J2n X = J2p, where X^\dag = J2p' X' J2n

## References
[Bin Gao](https://www.gaobin.cc/), [Nguyen Thanh Son](https://sites.google.com/view/ntson), [P.-A. Absil](https://sites.uclouvain.be/absil/), [Tatjana Stykel](https://www.uni-augsburg.de/en/fakultaet/mntf/math/prof/numa/team/tatjana-stykel/)
1. [Riemannian optimization on the symplectic Stiefel manifold](https://arxiv.org/abs/2006.15226)
2. Riemannian gradient method on the symplectic Stiefel manifold based on the Euclidean metric

