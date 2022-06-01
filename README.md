Python implementation of SpOpt package (originally in Matlab): Reimannian Optimization on Symplectic Stiefel Manifold 

# PySpOpt
A Python solver for Riemannian Optimization on the Symplectic Stiefel manifold. This was originally written in Matlab by [1].

## Problems
Solves the following optimization problem,

$$ \min f(X), \quad s.t. \quad   X^{\top} J_{2n} X = J_{2p}, $$

  
where $X$ is a 2n-by-2p matrix,
$J_{2n} = \begin{bmatrix} 0 & I_{n} \\\ - I_{n} & 0 \end{bmatrix}$,
and $\mathbb{I}_{n}$ is the n-by-n identity matrix.

## Applications
1. The nearest symplectic matrix problem:

$$ \min \Vert X-A \Vert ^{2}_{F}, \quad \text{s.t.} \quad X^{\top} J_{2n} X = J_{2p}. $$

2. The extrinsic mean problem:
  
$$ \min \frac{1}{N} \sum_{i=1}^{i=N} \Vert X - A_{i} \Vert^{2}_{\mathrm{F}},\quad \text{s.t.}\quad  X^{\top} J_{2n} X = J_{2p}. $$
  
3. Minimization of the Brockett cost function:

$$ \min \mathrm{Tr}(X^{\top} A X - 2 B X^{\top}),\quad \text{s.t.}\quad  X^{\top} J_{2n} X = J_{2p}. $$
  
4. Symplectic eigenvalue problem:

$$ \min \mathrm{Tr}(X^{\top} A X),\quad \text{s.t.}\quad  X^{\top} J_{2n} X = J_{2p}. $$
  
5. Symplectic model order reduction:

$$ \min \Vert M- X X^{\dagger} M \Vert, \quad \text{s.t.} \quad  X^{\top} J_{2n} X = J_{2p}, \quad \text{where} \quad X^{\dagger} = J_{2p}^{\top} X^{\top} J_{2n} .$$

## References
[Bin Gao](https://www.gaobin.cc/), [Nguyen Thanh Son](https://sites.google.com/view/ntson), [P.-A. Absil](https://sites.uclouvain.be/absil/), [Tatjana Stykel](https://www.uni-augsburg.de/en/fakultaet/mntf/math/prof/numa/team/tatjana-stykel/)
1. [Riemannian optimization on the symplectic Stiefel manifold](https://arxiv.org/abs/2006.15226)
2. Riemannian gradient method on the symplectic Stiefel manifold based on the Euclidean metric

