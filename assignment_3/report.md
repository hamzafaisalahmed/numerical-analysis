Problem 1: Lagrange Interpolation

i: Kronecker property, nodes x = [0, 1.5, 2, 4.2]
- max |Lk(x_j) − delta_kj| = 0 
- max |summation Lk(x) − 1|  = 1.1546319456101628e-14

Why summation Lk(x) = 1: it is the interpolating polynomial of the data y_k = 1 for all k. The constant polynomial 1 has degree <= 3 and matches that data, and the interpolant of degree <= n is unique, so summation Lk(x) ≡ 1.



ii: Exactness, f(x) = 2x³ − x + 5
Nodes: −1.3, 0.2, 1.1, 2.7 
Maximum error = 1.4210854715202004e-14

Why exact it reproduce this f: f has degree 3 and four nodes are used. then 4th derivative ≡ 0, so the error term f(n+1th derivative)(epsilon)/(n+1)! · Π(x − x_i) is zero. so f itself is unique P3



iii: Runge phenomenon, f(x) = 1/(1 + 25x²) on [−1, 1]
Max error on a 2001-point grid:

| n  | Equispaced nodes | Chebyshev nodes |
|----|------------------|-----------------|
5  | 0.4326923076923078  | 0.5559113388123953
10 | 1.915643050219248  | 0.10915326641231027
15 | 2.1075518461337115 | 0.0831070477847462
20 | 59.8223087107281   | 0.015332917318154893

node coincidence tolerance: Two nodes are treated as coinciding if |xi − xj| <= 10^-12 x max(|xi|, |xj|)



problem1 ai: help in generating if main block for testing and docstrings for functions 30%

