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



=====================================================================================================

PROBLEM 2: Newton forward and backward differences

=== Difference table of e^x ===
           x        f(x)          Δy        Δ^2y        Δ^3y        Δ^4y        Δ^5y
------------------------------------------------------------------------------------
    0.000000    1.000000    0.221403    0.049019    0.010853    0.002403    0.000532
    0.200000    1.221403    0.270422    0.059872    0.013256    0.002935            
    0.400000    1.491825    0.330294    0.073128    0.016191                        
    0.600000    1.822119    0.403422    0.089319                                    
    0.800000    2.225541    0.492741                                                
    1.000000    2.718282         


Test (i), accuracy vs degree 
-------------------------------------------------------------------------------------
Newton forward , e^0.05  (true = 1.05127110)
 deg         value    true error     error_est   base
   1    1.05535069     4.080e-03     4.596e-03   0.00
   2    1.05075514     5.160e-04     5.935e-04   0.00
   3    1.05134866     7.757e-05     9.034e-05   0.00
   4    1.05125832     1.278e-05     1.500e-05   0.00

Newton backward, e^0.95  (true = 2.58570966)
 deg         value    true error     error_est   base
   1    2.59509660     9.387e-03     8.374e-03   1.00
   2    2.58672297     1.013e-03     8.854e-04   1.00
   3    2.58583754     1.279e-04     1.103e-04   1.00
   4    2.58572719     1.753e-05     1.500e-05   1.00

The error drops by roughly 5 to 10 times per degree.
Forward is accurate near the start of the table and backward near the end, which is why each point uses the matching formula.
The estimate (first omitted term) is the same order of magnitude as the true error in every case. It slightly overestimates for forward and slightly underestimates for backward at degrees 1 to 3, so it is a guide, not a strict bound.

----------------------------------------------------------------------------------------------

Test (ii), same polynomial, two formulas.
-----------------------------------------------------------------------------------------------
all 6 nodes : max|forward - backward| = 8.882e-16
degree = 2  : max|forward - backward| = 2.023e-03
-----------------------------------------------------------------------------------------------


there is exactly one polynomial of degree at most 5 through 6 distinct points. Forward and backward are two ways of writing it, so they agree up to round-off.

forward uses three nodes at the start of the table and backward uses three at the end, so they are two different quadratics that approximate the same function.

with degree=2 both formulas have the same max true error (about 1.18e-03) but differ from each other by 2.02e-03. That shows they are two different valid approximations.

-----------------------------------------------------------------------------------------------
====== Design Choices =======

nodes are equally spaced if every step differs from h by at most 1e-9 · h. It is relative, 
so it works at any scale. Too tight rejects data with decimal round-off, and too loose lets uneven data through, which makes the formulas wrong.

Base node: forward uses the largest node <=  x_eval, and backward uses the smallest node >= x_eval. If degree+1 nodes don’t fit, the base node is shifted inward. With degree=None, all nodes are used (k = 0 forward, k = n-1 backward).

Difference table: both functions call difference_table once. Backward differences come from the same table.

Error estimate: the magnitude of the first omitted term, or None if no higher difference exists.

Array input: an array x_eval is evaluated point by point and returns an array of values.
-----------------------------------------------------------------------------------------------


==== Limitations =====

Nodes must be in ascending order.
Slightly uneven data is rejected, with no automatic fallback to Lagrange.
For array input, only values are returned, not terms or error estimates.
There is no extrapolation warning.
High degrees amplify round-off, and the error estimate assumes later terms are smaller.

-----------------------------------------------------------------------------------------------

===== AI USAGE ======

Tool Used : Cluade
Reviewed my forward/backward functions and pointed out bugs (indentation, degree check, base-node initialization), explained how to fix them, provided the __main__ test block, and drafted the Design Choices and Limitations text. I wrote and debugged the final functions myself. 

Estimate AI Usage: 30% 

-------------------------------------------------------------------------------------------------

=====================================================================================================