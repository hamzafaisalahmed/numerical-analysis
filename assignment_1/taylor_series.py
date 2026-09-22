import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def taylor_series(expr, point, n, var=None):
    """
    Construct the n-th order Taylor polynomial of 'expr' about 'point',
    symbolically, using term-by-term differentiation.

    Parameters
    ----------
    expr : sympy.Expr or str
        The function to expand, e.g. sympy.exp(x) or "exp(x)*sin(x)".
    point : int, float, or sympy number
        The expansion point x_0.
    n : int
        The order of the Taylor polynomial (must satisfy n >= 0).
    var : sympy.Symbol, optional
        The variable of expansion. If None, its inferred
        automatically from 'expr' (raises an error if 'expr' has zero
        or more than one free symbol and var was not supplied).

    Returns
    -------
    dict
        {
            "polynomial": sympy.Expr,  # P_n(x), fully simplified
            "terms": list[sympy.Expr], # individual terms, term[k] is the kth-order contribution
            "remainder": sympy.Expr,   # symbolic R_n(x) with a fresh symbol xi
            "latex": str               # LaTeX string of the polynomial
        }

    Raises
    ------
    TypeError
        If expr cannot be converted to a valid SymPy expression or is not a SymPy Expr instance.
    ValueError
        If n < 0 , if 'var' cannot be inferred , or if 'expr' is not n+1 times differentiable at a symbolic level 
        (e.g. sympy cannot compute the derivative ).


    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> result = taylor_series(sp.exp(x), 0, 2)
    >>> result["polynomial"]
    x**2/2 + x + 1
    """

    expr = sp.sympify(expr)
    if not isinstance(expr, sp.Expr):
        raise TypeError("expr must be a valid sympy expression")
    if var is None:
        var = find_var(expr)
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non negative integer")

    return_expr = 0
    return_data = {
        "polynomial": return_expr,
        "terms": [],
        "remainder": return_expr,
        "latex": "",
    }

    for i in range(n + 1):
        derivative = expr.diff(var, i).subs(var, point)

        if derivative == sp.nan:
            raise ValueError(
                f"derivative not defined at {point} for n = {n}, function is not differentiable at {point}"
            )
        elif derivative == sp.zoo:
            raise ValueError(
                f"derivative not defined at {point} for n = {n}, limit goes to infinity"
            )

        term = derivative / sp.factorial(i) * (var - point) ** (i)
        return_expr += term
        return_data["terms"].append(term)
    return_data["polynomial"] = sp.expand(return_expr)
    return_data["latex"] = sp.latex(return_expr)

    xi = sp.symbols("xi")
    return_data["remainder"] = (
        expr.diff(var, n + 1).subs(var, xi) / sp.factorial(n + 1)
    ) * (var - point) ** (n + 1)

    return return_data


# |------------------------------ Helper Functions ------------------------------|
def find_var(expr):
    """
    Helper function to find the variable of expansion in a sympy expression.

    Parameters
    ----------
    expr : sympy.Expr
        The expression from which to infer the variable.
    Returns
    -------
    sympy.Symbol
        The variable of expansion.
    Raises
    ------
    ValueError
        If 'expr' has zero or more than one free symbol.
    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> expr = sp.exp(x) + sp.sin(x)
    >>> find_var(expr)
    x
    """
    variables = expr.free_symbols
    if len(variables) == 0:
        raise ValueError("unable to infer variable: 0 free symbols")
    if len(variables) > 1:
        raise ValueError("unable to infer variable: > 1 free symbols")
    return next(iter(variables))


def evaluate_and_compare(expr, point, n, x_eval, var=None):
    """
    Evaluates a Taylor polynomial and the function's value at a specific point to compute errors.

    Parameters
    ----------
    expr : sympy.Expr or str
        The function to approximate.
    point : float or int
        The expansion point x_0 for the Taylor series.
    n : int
        The order of the Taylor polynomial.
    x_eval : float
        The point at which to evaluate both the function and the approximation.
    var : sympy.Symbol, optional
        The variable of expansion. Inferred automatically if None.

    Returns
    -------
    dict
        A dictionary containing:
        - "approx": The numerical value of the Taylor polynomial at x_eval.
        - "true_value": The numerical value of the actual function at x_eval.
        - "abs_error": The absolute difference between true_value and approx.
        - "rel_error": The relative error of the approximation.

    Example
    -------
    >>> evaluate_and_compare('exp(x)', 0, 3, 0.5)
    {'approx': 1.64583333333333, 'true_value': 1.64872127070013, 'abs_error': 0.00288793736679493, 'rel_error': 0.00175162255629090}
    """

    if type(expr) == str:
        expr = sp.sympify(expr)
    if var == None:
        var = find_var(expr)
    series = taylor_series(expr, point, n, var)

    approx = sp.N(series["polynomial"].subs(var, x_eval))
    true_value = sp.N(expr.subs(var, x_eval))
    abs_error = abs(true_value - approx)
    rel_error = abs_error / abs(true_value)
    return {
        "approx": approx,
        "true_value": true_value,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def plot_taylor_approximations(expr, point, orders, x_range, var=None):
    """
    Plots the true function 'expr' alongside its Taylor polynomial approximations
    for various orders over a specified range using matplotlib.

    Parameters
    ----------
    expr : sympy.Expr or str
        The function to approximate and plot.
    point : int, float, or sympy number
        The expansion point x_0 for the Taylor series.
    orders : list[int]
        A list of polynomial orders to plot (e.g., [1, 2, 4, 6]).
    x_range : list[float] or tuple[float, float]
        The lower and upper bounds for the x-axis domain [xmin, xmax].
    var : sympy.Symbol, optional
        The variable of expansion. Inferred automatically from 'expr' if None.

    Returns
    -------
    None
        Saves the resulting plot as 'taylor_plot.png' to the current directory.

    Raises
    ------
    ValueError
        If 'var' cannot be inferred or if orders contain invalid values.

    Example
    -------
    >>> import sympy as sp
    >>> x = sp.Symbol('x')
    >>> plot_taylor_approximations(sp.cos(x), 0, [1, 2, 4, 6], [-6.28, 6.28])
    """

    if type(expr) == str:
        expr = sp.sympify(expr)

    if type(orders) == int:
        orders = [orders]
    elif type(orders) != list:
        raise ValueError("orders must be an integer or a list of integers")
    x_arr = np.linspace(x_range[0], x_range[1], 300)
    if var == None:
        var = find_var(expr)
    f = sp.lambdify(var, expr)
    y_f = f(x_arr)
    if np.isscalar(
        y_f
    ):  # to avoid value error incase the whole array is shrunk to one value
        y_f = np.full_like(x_arr, y_f)

    plt.plot(x_arr, y_f, label="f(x)")
    for order in orders:
        series = taylor_series(expr, point, order, var)
        der = sp.lambdify(var, series["polynomial"])
        y_arr = der(x_arr)
        label = f"P_{order}(x)"

        if np.isscalar(y_arr):
            y_arr = np.full_like(x_arr, y_arr)

        plt.plot(x_arr, y_arr, label=label)
    plt.legend()
    plt.title(f"Taylor Approximations of ${sp.latex(expr)}$ at x={point}")
    plt.savefig("taylor_plot.png", dpi=300)


if __name__ == "__main__":
    x = sp.Symbol("x")

    # test case i
    print("Test case i: ")
    expr = sp.exp(x)
    x0 = 0
    N = [1, 3, 5]
    print(f"For {expr}")
    for n in N:
        tc1 = evaluate_and_compare(expr, x0, n, 0.5)
        print(
            f"at n = {n}: Absolute error = {tc1['abs_error']}, Relative error = {tc1['rel_error']}"
        )

    # test case ii
    print("\nTest case ii: ")
    expr = sp.sin(x)
    x0 = 0
    n = 7
    tc2 = taylor_series(expr, x0, n)
    print(f"\nFor {expr} at n = {n}:")
    print(f"Terms: {tc2['terms']}")
    print("(all even order terms vanished)")

    # test case iii
    print("\nTest case iii: ")
    expr = sp.ln(x)
    x0 = 1
    n = 4
    tc3 = evaluate_and_compare(expr, x0, n, 1.2)
    print(f"\nFor {expr} at n = {n}:")
    print(f"P_4(1.2) = {tc3['approx']}, ln(1.2) = {tc3['true_value']}")
    print(f"Absolute error = {tc3['abs_error']}, Relative error = {tc3['rel_error']}")

    # test case iv
    print("\nTest case iv: ")
    expr = 1 / (1 - x)
    x0 = 0
    n = 5
    xin = [0.9, 1.5]
    tc4_1 = evaluate_and_compare(expr, x0, n, xin[0])
    tc4_2 = evaluate_and_compare(expr, x0, n, xin[1])
    print(f"\nFor {expr} at n = {n}:")
    print(f"P_5(0.9) = {tc4_1['approx']}, 1/(1-{xin[0]}) = {tc4_1['true_value']}")
    print(
        f"Absolute error = {tc4_1['abs_error']}, Relative error = {tc4_1['rel_error']}"
    )
    print(f"P_5(1.5) = {tc4_2['approx']}, 1/(1-{xin[1]}) = {tc4_2['true_value']}")
    print(
        f"Absolute error = {tc4_2['abs_error']}, Relative error = {tc4_2['rel_error']}"
    )

    # test case v
    # tc 5.1
    print("\nTest case 5.1: ")
    expr = x**2
    n = 0
    x0 = x
    tc5_1 = taylor_series(expr, x0, 0, x)
    print(f"When n = 0, derivative of {expr} is still {tc5_1['polynomial']}")

    x0 = 2
    tc5_1 = taylor_series(expr, x0, 0, x)
    print(f"When n = 0, {expr} at x = {x0} is still {tc5_1['polynomial']}")

    # tc 5.2
    print("\nTest case 5.2: ")
    try:
        tc5_1 = taylor_series(expr, x0, -1, x)

    except ValueError as msg:
        print(f"Error: {msg}")
    try:
        tc5_1 = taylor_series(expr, x0, "a", x)
    except ValueError as msg:
        print(f"Error: {msg}")

    # tc 5.3
    print("\nTest case 5.3: ")
    try:
        expr = sp.Integer(5)
        taylor_series(expr, x0, 3)
    except ValueError as msg:
        print(f"Error: {msg}")
    # tc 5.4
    print("\nTest case 5.4: ")
    try:
        y = sp.Symbol("y")
        expr = x * y
        taylor_series(expr, x0, 2)
    except ValueError as msg:
        print(f"Error: {msg}")

    # tc 5.5
    print("\nTest case 5.5: ")
    try:
        expr = sp.Abs(x)
        taylor_series(expr, 0, 1)
    except ValueError as msg:
        print(f"Error: {msg}")
    try:
        expr = sp.sqrt(x)
        taylor_series(expr, 0, 1)
    except ValueError as msg:
        print(f"Error: {msg}")

    # tc 5.6
    print("\nTest case 5.6: ")
    try:
        expr = sp.ln(x)
        x0 = 1
        tc_5_6 = taylor_series(expr, x0, 1)
        print(f"Polynomial for lnx for x0 = {x0} is {tc_5_6['polynomial']}")
        x0 = sp.Symbol("x0")
        tc_5_6 = taylor_series(expr, x0, 1)
        print(f"Polynomial for lnx for x0 = {x0} is {tc_5_6['polynomial']}")
    except ValueError as msg:
        print(f"Error: {msg}")

    # test case vi
    print("\nTest case vi: (image saved as taylor_plot.png)")
    expr = sp.cos(x)
    range_vi = float(2 * np.pi)
    plot_taylor_approximations(expr, 0, [1, 2, 4, 6], [-range_vi, range_vi])
