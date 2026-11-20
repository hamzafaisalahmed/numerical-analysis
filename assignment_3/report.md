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

Test (ii), same polynomial, two formulas
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

Array input: an array x_eval is evaluated point by point and returns an array of values
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

=====================================================================================================

PROBLEM 3: Hermite interpolation

Test (i): is derivative information worth it? f(x) = e^x on [0, 1]

| Method                    | Nodes | Error at x = 0.25 | Max error on fine grid |
|---------------------------|-------|-------------------|------------------------|
| Lagrange, 3 nodes (deg 2) | 3     | 1.227e-02         | 1.442e-02              |
| Hermite, 3 nodes (deg 5)  | 3     | 4.901e-06         | 5.577e-06              |
| Lagrange, 6 nodes (deg 5) | 6     | 6.022e-07         | 2.655e-06              |

Fewest measurement locations: Hermite (3 nodes, each with a value and a slope).
Most accurate: Lagrange with 6 nodes, by about 8x at x = 0.25 and about 2x on the grid.
Hermite with 3 nodes is about 2500x better than Lagrange with 3 nodes at x = 0.25, and
it gets within a small factor of 6-node Lagrange using half the locations.

Explanation with the error formulas (both have a 1/6! factor and a degree-5 polynomial):

- Lagrange 6 nodes: f(x) - P5(x) = f^(6)(xi)/6! * prod_{i=0..5} (x - x_i)
- Hermite 3 nodes:  f(x) - H5(x) = f^(6)(xi)/6! * prod_{i=0..2} (x - x_i)^2
At x = 0.25 the Lagrange product is 2.7e-4 and the Hermite product is 2.2e-3, about 8 times
larger, which matches the ratio of the observed errors (6.0e-7 vs 4.9e-6). The Hermite
product is larger because it has only 3 distinct nodes, so the factors (x - x_i)^2 are
less well spread over the interval. The Lagrange 3-node error has only a 1/3! factor
and a degree-3 product, so it is far worse.

Test (ii): interpolation conditions (data from (i)(b))
max |H(x_i) - f(x_i)|   = 0.0
max |H'(x_i) - f'(x_i)| = 0.0
Both hold at every node (to round-off or better).

Test (iii): exactness with 3 nodes (n = 2, degree 2n+1 = 5)
max error for x^5 - 3x^2 + 1 = 6.66e-16
max error for x^6            = 2.31e-03
actual error at x = 0.25     = 0.002197265625
prod (x - x_i)^2 at x = 0.25 = 0.002197265625

Why: p(x) = x^5 - 3x^2 + 1 has degree 5 <= 2n+1, so f^(6) = 0 and the error term is zero.
The interpolant is unique, so it equals p. For x^6, f^(6) = 720, so f^(6)/6! = 1 and
the error is exactly prod (x - x_i)^2, which is not zero away from the nodes. The two
numbers above agree, which confirms the error formula.

----- Design Decisions (Problem 3) -----

Doubled nodes: z = [x0, x0, x1, x1, ...]. In the divided-difference table, the first-level
entry for a repeated node uses f'(x_i) instead of 0/0. Nodes are not sorted and may be in
any order.

H'(x): computed analytically inside the Horner loop. At each step the product rule gives
dH_new = dH_old * (x - z_i) + H_old, so H' is found alongside H with no finite differences.

Validation: x and y reuse check_data from lagrange.py (node coincidence tolerance is
1e-12 relative). dydx_nodes is checked for numeric type, matching length, and nan/inf.

Extrapolation: a warning is raised through the warnings module if x_eval is outside
[min(x_nodes), max(x_nodes)].

----- Limitations (Problem 3) -----

Needs derivative values at every node; it cannot use derivatives at only some of them.
Inaccurate derivative data gives a wrong polynomial, because the slopes are matched exactly.
High degree amplifies round-off, and nodes that are very close make the divided differences
large (division by small gaps).
x_eval must be a scalar or a 1-D array.
The coefficient table is built with O(n^2) Python loops, which is fine for small data sets.

----- AI Usage (Problem 3) -----

Tool used: Claude
wrote the __main__ test block
wrote the report.md

Estimate AI usage:
30%
--------------------------------------------------------------------------------------------------

PROBLEM 4: Automatic method selection

----- Required test cases (y = sin(x); f'(x) = cos(x) for T4) -----

| Case | x-nodes                  | Extra data  | x_eval | Predicted method | Chosen method   | Value       | sin(x_eval) | Abs. error |
|------|--------------------------|-------------|--------|------------------|-----------------|-------------|-------------|------------|
| T1   | 0,1,2,3,4,5              | none        | 0.4    | forward          | newton_forward  | 0.37401147  | 0.38941834  | 1.5e-02    |
| T2   | 0,1,2,3,4,5              | none        | 4.7    | backward         | newton_backward | -1.00595938 | -0.99992326 | 6.0e-03    |
| T3   | 0,0.3,1.1,2.0,3.5        | none        | 1.5    | lagrange         | lagrange        | 0.99653922  | 0.99749499  | 9.6e-04    |
| T4   | 0,0.3,1.1,2.0,3.5        | f'(x_i)     | 1.5    | hermite          | hermite         | 0.99749512  | 0.99749499  | 1.3e-07    |
| T5   | 5,4,3,2,1,0 (unsorted)   | none        | 0.4    | forward (sorted) | newton_forward  | 0.37401147  | 0.38941834  | 1.5e-02    |
| T6   | 0,0.1,0.2000000001,0.3   | none        | 0.15   | forward          | newton_forward  | 0.14943778  | 0.14943813  | 3.5e-07    |
| T7   | 0,1,2,3,4,5              | none        | 6.5    | backward         | newton_backward | 0.38419933  | 0.21511999  | 1.7e-01    |

Prediction vs program: they agree in all seven cases.

Comments on individual cases:

- T5: gives exactly the same value as T1. The reversed nodes are sorted
  together with y, so every x_i stays paired with its y_i. Without sorting
  the Newton formulas would use x_0 = 5 and give a wrong answer.
- T4: the most accurate result (error 1.3e-07). Hermite matches values and
  slopes, so with 5 nodes its degree is 9, against degree 4 for Lagrange in T3.
- T6: the gaps differ from h = 0.1 by about 1e-10, which is right at the
  tolerance 1e-9 * h. It passed the test (h = 0.1), so the nodes were treated
  as equally spaced. The forward/backward choice is not borderline (s = 1.5 vs
  n/2 = 2). If the test had failed, Lagrange would have been used and the value
  would be almost the same.
- T7: x_eval = 6.5 is outside [0, 5], so extrapolating = True (the only case).
  The error (1.7e-01) is much larger than for the interior points, because
  polynomial error grows quickly outside the nodes.
- T1, T2, T5: errors of about 1e-2 are due to the coarse spacing (h = 1) for an
  oscillating function, not to the choice of method.

----- Questions -----

1. Lagrange and Newton forward/backward give the same polynomial when all nodes
   are used. Why choose?

- Truncation: with a low degree we want the nodes nearest x_eval. Forward
  differences start from the beginning of the table and backward differences
  from the end, so each gives a smaller error near its own end. Lagrange has
  no such advantage.
- Error estimates: the next difference term in Newton's formula is a free
  estimate of the truncation error. Lagrange gives none without extra work.
- Adding new data: Newton appends one new term. Lagrange has to recompute
  every basis polynomial.
- Cost per evaluation: once the difference table is built, Newton is O(n)
  per point. Lagrange is O(n^2) unless barycentric weights are precomputed.
- Lagrange works for any spacing, so it is the general fallback.

1. Why is rule 1 (derivatives available -> Hermite) first?
Derivative values are extra information, and ignoring them wastes it. Hermite
matches both values and slopes, so it is more accurate for the same number of
nodes (T4 vs T3). It also works for any spacing, so there is no need to test
the spacing before choosing it.

2. T6: is it correct to treat the data as equally spaced?
Strictly, no. The gaps are 0.1, 0.1000000001 and 0.0999999999. But a relative
deviation of 1e-9 is far smaller than the interpolation error, so treating it
as equal is reasonable, and the result (error 3.5e-07) shows no harm.

- Tolerance too tight (e.g. 0): round-off alone makes ordinary data such as
  0.1, 0.2, 0.3 look unequal, so everything falls back to Lagrange and the
  cheaper Newton formulas are never used.
- Tolerance too loose: genuinely uneven data is accepted as equal, h is
  wrong, and the Newton formulas give wrong answers without any error.

----- Design Choices (Problem 4) -----

sort_inputs: sorts x, y and dy/dx together with one argsort index, so the
pairs stay matched. Called inside both interpolate and choose_method, so each
is correct when called on its own.

Spacing: reuses check_equal_spacing from the Newton file (relative tolerance
1e-9). A ValueError from it means "not equally spaced". The check also runs
when derivatives are given, so equally_spaced and h are reported truthfully
for Hermite.

n and the halfway point: n = len(x) (number of nodes), so the test is
s <= n/2. This only matters at the halfway boundary and does not change any
of T1-T7.

y_nodes: does not affect the decision. It is only passed to sort_inputs, which
checks that the lengths match.

Extrapolating: True if x_eval is below min(x) or above max(x). It is computed
for every method, not only Newton.

Input checks: x_eval must be a single int or float (TypeError otherwise).
interpolate raises ValueError if choose_method returns an unknown method name.

verbose=True: prints method, value, degree, equally_spaced/h, extrapolating
and the reason, one block per call.

----- Limitations (Problem 4) -----

x_eval must be a single number, not an array. The type check accepts only
Python int and float (numpy integers are rejected, True/False are accepted,
and nan/inf are not checked).
spacing_rtol is fixed at 1e-9 and is not exposed through interpolate.
Slightly uneven data goes to Lagrange, even when Newton would be nearly exact.
Newton forward/backward warn nothing for extrapolation; the extrapolating
flag and verbose output are the only notice.

----- AI Usage (Problem 4) -----

Tool used: Claude
Explained the steps for choose_method and interpolate, and reviewed my code.
Bugs it pointed out: sort_inputs returned an empty array instead of None for
dy/dx, the appended list was the wrong one, n was used before it was defined,
a float was concatenated to a string, the argument order of the Hermite call,
and an off-by-one in the T4 index. It also helped with the verbose output
format, the docstrings and the wording of this report. I wrote and debugged
the final functions myself.

Estimate AI usage: 30%
--------------------------------------------------------------------------------------------------

PROBLEM 5: Real-World Applications

Application A: Population Between Census Years (Demography)

Method chosen by choose_method:
- x_eval = 1955 -> newton_forward (h = 10, s = 0.5 of n = 8)
- x_eval = 2015 -> newton_backward (h = 10, s = 6.5 of n = 8)
Agreement: yes. The census years are equally spaced (h = 10) with no derivatives,
1955 is near the beginning of the table, and 2015 is near the end.

(a) Forward difference table:
           x        f(x)          Δy        Δ^2y        Δ^3y        Δ^4y        Δ^5y        Δ^6y        Δ^7y
------------------------------------------------------------------------------------------------------------
   1950.0000    151.3000     28.0000     -4.0000      3.2000     -3.4000     15.1000    -54.2000    137.4000
   1960.0000    179.3000     24.0000     -0.8000     -0.2000     11.7000    -39.1000     83.2000            
   1970.0000    203.3000     23.2000     -1.0000     11.5000    -27.4000     44.1000                        
   1980.0000    226.5000     22.2000     10.5000    -15.9000     16.7000                                    
   1990.0000    248.7000     32.7000     -5.4000      0.8000                                                
   2000.0000    281.4000     27.3000     -4.6000                                                            
   2010.0000    308.7000     22.7000                                                                        
   2020.0000    331.4000                                                                                    

The higher differences are irregular, not small and smooth like e^x in Problem 2.
From Δ^3y onward they flip sign and grow large (up to 137.4 at Δ^7y), because
census counts are affected by baby booms, immigration changes, and rounding noise
rather than following one smooth formula. This tells us a low degree (degree 2 or 3)
using nearby nodes is sensible; a high degree would amplify noise and oscillate.

(b) Degree-3 estimates:
- 1955 (newton_forward, degree = 3, base_node = 1950, s = 0.5):
  value = 166.0000 million, terms = [151.3, 14.0, 0.5, 0.2], error_estimate = 0.1328 million
- 2015 (newton_backward, degree = 3, base_node = 2020, t = -0.5):
  value = 320.5750 million, terms = [331.4, -11.35, 0.575, -0.05], error_estimate = 0.6523 million

Application B: Reading Between the Lines of a Steam Table (Thermodynamics) 

Method chosen by choose_method:
- T = 25 °C -> newton_forward (h = 10, s = 0.5 of n = 5)
- T = 55 °C -> newton_backward (h = 10, s = 3.5 of n = 5)
Agreement: yes. The temperatures are equally spaced (h = 10 °C) with no derivatives,
25 °C lies in the first half of the table, and 55 °C lies in the second half.

(a) & (b) Estimates, relative errors, and error_estimate:

T = 25 °C (reference P_sat = 3.1698 kPa, method = newton_forward, base = 20 °C):

| Degree | P_sat (kPa) | Abs. error | Rel. error (%) | error_estimate |
|--------|-------------|------------|----------------|----------------|
| 1      | 3.293050    | 1.2325e-01 | 3.8883%        | 1.5381e-01     |
| 2      | 3.139238    | 3.0562e-02 | 0.9642%        | 3.7387e-02     |
| 3      | 3.176625    | 6.8250e-03 | 0.2153%        | 7.8594e-03     |
| 4      | 3.168766    | 1.0344e-03 | 0.0326%        | None           |

T = 55 °C (reference P_sat = 15.7630 kPa, method = newton_backward, base = 60 °C):

| Degree | P_sat (kPa) | Abs. error | Rel. error (%) | error_estimate |
|--------|-------------|------------|----------------|----------------|
| 1      | 16.149500   | 3.8650e-01 | 2.4519%        | 3.2851e-01     |
| 2      | 15.820987   | 5.7987e-02 | 0.3679%        | 4.9962e-02     |
| 3      | 15.771025   | 8.0250e-03 | 0.0509%        | 7.8594e-03     |
| 4      | 15.763166   | 1.6562e-04 | 0.0011%        | None           |

(c) Shape of curve, safety, and recommendation (plot saved as steam_table.png):
- Why linear interpolation overestimates: P_sat(T) grows roughly exponentially with T,
  so the curve is concave up (convex, P''(T) > 0) everywhere. Any straight secant line
  connecting two points on a concave-up curve sits above the curve.
- Safer error for pressure vessel design: when pressure determines the required wall
  thickness, an overestimate is the safer error because it gives a slightly thicker,
  stronger wall, whereas an underestimate could produce a wall too thin for the true
  operating pressure.
- Recommendation: 4 terms (degree 3) is a good engineering choice—it reduces relative
  error to 0.22% at 25 °C and 0.05% at 55 °C while still leaving the 5th point to give
  an error_estimate (or all 5 terms / degree 4 if maximum accuracy < 0.033% is needed).


Application C: Drug Concentration in the Blood (Pharmacokinetics) 

Method chosen by choose_method:
- t = 3 h -> lagrange
Agreement: yes. The sample times are unequally spaced and no derivatives are given.

(a) Why neither Newton forward nor backward can be used:
The time steps between blood draws are 0.5, 0.5, 1, 2, 2, 2, and 4 h. Because the spacing
is not constant, finite differences are not defined on a uniform step h and
check_equal_spacing rejects the data.

(b) Estimate at t = 3 h (degree-7 Lagrange through all 8 points):
C(3 h) = 5.669397 mg/L

(c) Global degree-7 plot on [0, 12] with 2000 points (saved as pharmacokinetics.png):
- Predicts negative concentration: yes, for t in [8.7104 h, 11.9760 h] (between the last
  two nodes t = 8 h and t = 12 h), reaching a minimum of -11.2572 mg/L at t = 11.0155 h.
- Why impossible and lesson: drug concentration cannot be negative (there cannot be less
  than zero drug in the blood). Fitting one high-degree polynomial blindly through all
  points causes large Runge-like swings across wide gaps ([8, 12] h). A low-degree local
  interpolant around the region of interest avoids these unphysical swings.

(d) Peak concentration C_max and time to peak t_max:
- Global degree-7 interpolant: C_max = 6.6101 mg/L at t_max = 1.6568 h
- Local degree-3 interpolant (nodes t = 0.5, 1, 2, 4 h): C_max = 6.7062 mg/L at t_max = 1.6171 h


Application D: Was the Car Speeding? (Traffic Enforcement — Hermite)

Method chosen by choose_method:
- hermite (derivatives given)
Agreement: yes. Both position s(t_i) and speed v(t_i) = s'(t_i) are recorded at every checkpoint.

(a) Recorded speeds and average speeds vs. speed limit (100 km/h = 27.7778 m/s):
- Recorded checkpoint speeds: 24.5, 25.0, 24.0, 25.5, 24.0 m/s (all <= 25.5 m/s < 27.78 m/s).
- Average speeds between consecutive checkpoints:
  [0, 12] s  : 312 / 12 = 26.0000 m/s (93.60 km/h)
  [12, 25] s : 343 / 13 = 26.3846 m/s (94.98 km/h)
  [25, 40] s : 407 / 15 = 27.1333 m/s (97.68 km/h)
  [40, 52] s : 316 / 12 = 26.3333 m/s (94.80 km/h)
Every recorded speed and every average speed is below 27.78 m/s. Based only on these
numbers, the driver would not be fined.

(b) Which interpolation method this data calls for and why:
Hermite interpolation (degree 9 on 5 nodes), because both the function value s(t_i)
and its derivative v(t_i) = s'(t_i) are known at each unequally spaced checkpoint.
Lagrange would ignore the speed measurements, and Newton forward/backward cannot use
unequally spaced times or derivatives.

(c) Hermite (degree 9) estimates at t = 18, 33, 46 s (plot saved as speed_profile.png):

| t (s) | Position H(t) (m) | Speed H'(t) (m/s) | Speed (km/h) | Exceeds 27.78 m/s? |
|-------|-------------------|-------------------|--------------|--------------------|
| 18.0  | 475.1553          | 27.7799           | 100.01       | Yes                |
| 33.0  | 862.5932          | 28.9689           | 104.29       | Yes                |
| 46.0  | 1201.9417         | 24.4854           | 88.15        | No                 |

In every interval, the checkpoint speeds (24.0 to 25.5 m/s) are below the interval's
average speed (26.00 to 27.13 m/s), which means the car slowed down at each camera and
sped up between cameras. H'(t) exceeds 27.78 m/s in all four intervals ([0.44, 4.29] s,
[15.37, 17.99] s, [31.43, 38.14] s, and [47.48, 51.56] s), reaching 28.97 m/s
(104.29 km/h) at t = 33 s and a peak of 32.60 m/s (117.35 km/h) at t = 50.00 s.


----- Method-Selection Summary -----

| Step | Condition to check                         | Method to use   | Why                                                      |
|------|--------------------------------------------|-----------------|----------------------------------------------------------|
| 1    | Derivatives f'(x_i) are given at the nodes | Hermite         | Matches both values and slopes (degree 2n+1)             |
| 2a   | Equally spaced (h constant) and s <= n/2   | Newton forward  | Uses differences near the start of the table; O(n) eval  |
| 2b   | Equally spaced (h constant) and s > n/2    | Newton backward | Uses differences near the end of the table; O(n) eval    |
| 3    | Unequally spaced and no derivatives        | Lagrange        | Works for any distinct node spacing                      |


----- Design Choices & Limitations (Problem 5) -----

Design choices:
- Each application has a reusable evaluation function that accepts arbitrary data
  (x_nodes, y_nodes, optional dydx_nodes) and scalar or 1-D array x_eval, plus a
  runner function called from __main__.
- Application B delegates forward/backward selection to interpolate() for each degree.
- Application C compares the global degree-7 Lagrange polynomial with a local degree-3
  Lagrange polynomial on the four nodes surrounding the peak.
- Application D computes H(t) and H'(t) analytically in a single hermite_interpolate
  call with return_derivative=True.

Limitations:
- High-degree global polynomials on noisy (Application A) or wide/uneven (Application C
  and D) data exhibit Runge-like oscillations near the ends of the interval.
- interpolate() accepts scalar x_eval only, so array evaluation in Applications A and B
  loops over x_eval when calling interpolate(), while Applications C and D pass the
  1-D array directly to lagrange_interpolate and hermite_interpolate.

----- AI Usage (Problem 4) -----

Tool used: Gemini 4 Argon 

Used for generating doc strings, review of code fixing many minor and some major bugs.
Also used for generating the code for generating the images
Estimated Usage: 25%
