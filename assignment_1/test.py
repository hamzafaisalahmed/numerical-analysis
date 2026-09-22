import sympy as sp
import numpy as np

import bisection_method as bisec
import taylor_series as tay


#tests to get examples for docstrings in bisection_method.py and taylor_series.py
print("=" * 60)
print("VERIFYING BISECTION MODULE EXAMPLES")
print("=" * 60)

# 1. interval
a, b = 1.0, 3.5
print("interval(1.0, 3.5) ->", bisec.interval(a, b))

# 2. Residual
x = sp.Symbol('x')
expr_b = x**2 - 2
f_b = sp.lambdify(x, expr_b, 'numpy')
print("Residual(f, 1.5) ->", bisec.Residual(f_b, 1.5))

# 3. Absolute_error
p_prev, p_curr = 1.0, 1.5
print("Absolute_error(1.5, 1.0) ->", bisec.Absolute_error(p_curr, p_prev))

# 4. Relative_error
print("Relative_error(1.5, 1.0) ->", bisec.Relative_error(p_curr, p_prev))

# 5. ErorChecker
print("ErorChecker(...) ->", bisec.ErorChecker("interval", 1e-6, 5e-7, 0.1, 0.05, 0.03))

# 6. plot_convergence (matching primary test case f1 & res1)
f1 = lambda x: x**3 - x - 2
res1 = bisec.bisection(f1, 1.0, 2.0, tol=1e-6, error_type="interval")
bisec.plot_convergence(res1)


print("\n" + "=" * 60)
print("VERIFYING TAYLOR SERIES MODULE EXAMPLES")
print("=" * 60)

# 1. taylor_series
x_t = sp.Symbol('x')
res_taylor = tay.taylor_series(sp.exp(x_t), 0, 2)
print("taylor_series polynomial ->", res_taylor["polynomial"])

# 2. find_var
expr_v = sp.exp(x_t) + sp.sin(x_t)
print("find_var(...) ->", tay.find_var(expr_v))

# 3. evaluate_and_compare
print("evaluate_and_compare(...) ->", tay.evaluate_and_compare('exp(x)', 0, 3, 0.5))

# 4. plot_taylor_approximations
tay.plot_taylor_approximations(sp.cos(x_t), 0, [1, 2, 4, 6], [-2 * np.pi, 2 * np.pi])
print("Taylor plot generated and saved successfully.")