Problem 1 Report

Ranking behaviour:

f(x) = x^3 + 4x^2 - 10 = 0, starting from p0 = 1.5 with root p = 1.365230013:

1: g1(x) = x - x^3 - 4x^2 + 10 (Diverges)
   The derivative at the root is |g1'(p)| = 15.5134 which is > 1

2: g2(x) = sqrt(10/x - 4x) (Diverges)
   The derivative at the root is |g2'(p)| = 3.4299  which is > 1

3: g3(x) = 0.5 * sqrt(10 - x^3) (Converges slowly)
   The derivative at the root is |g3'(p)| = 0.5120, which is < 1

4: g4(x) = sqrt(10 / (4 + x)) (Converges fast)
   The derivative at the root is |g4'(p)| = 0.1272, which is < 1

5: g5(x) = x - (x^3 + 4x^2 - 10)/(3x^2 + 8x) (Converges very fast)
   The derivative at the root is |g5'(p)| = 0 


The rearrangement g5(x) is the same as Newtons Method:
g5(x) = x - f(x) / f'(x)
For the function f(x) = x^3 + 4x^2 - 10, the derivative is f'(x) = 3x^2 + 8x. Substituting these into Newton's formula gives exactly g5(x).
This is why g5 exhibits quadratic convergence since it follows newtons method 


Design decisions:
Error Floor: We use 1e-7 as a minimum error to avoid problems caused by very small rounding errors.
Successive Differences: When the true value is unknown, we use the difference between consecutive values. This is not the exact error, but gives a good estimate of the convergence order


Limitations:
cannot prevent stepping into invalid domains (like negative square roots in g2), but catches errors without crashing
if |g'(p)| is close to 1, convergence is very slow and can hit max iterations


AI Usage Declaration:

Tool used: chatgpt, gemini
Part of assignment: problem 1
Nature of assistance: generating docstrings of both functions, help in generating if __name__ == "__main__" block and code debugging (fixing syntax)
Estimated percentage AI-assisted: 15%
