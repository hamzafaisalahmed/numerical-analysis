import warnings

import numpy as np

from lagrange import check_data, lagrange_interpolate


def hermite_coefficients(x_nodes, y_nodes, dydx_nodes):
    """
    Build the doubled - node divided - difference table .

    Parameters
    ----------
    x_nodes, y_nodes, dydx_nodes : lists/arrays of equal length

    Returns
        -------
        dict : {" z ": ndarray of length 2( n +1) ,
        " coeffs ": ndarray of the diagonal entries
        f [ z0 ] , f [ z0 , z1 ] , ... , f [ z0 ,... , z_ {2 n +1}] ,
        " table ": the full 2 - D divided - difference table }

    Raises
    ------
    ValueError -- mismatched lengths, repeated nodes, nan/inf
    TypeError  -- non-numeric input

    Example
    -------
    >>> hermite_coefficients([0, 1], [0, 1], [0, 0])["coeffs"]
    array([ 0.,  0.,  1., -2.])
    """

    # simple boiler checking and exception handling
    x, y = check_data(x_nodes, y_nodes)
    assert y is not None
    try:
        slopes = np.asarray(dydx_nodes, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("dydx_nodes must be numeric.")
    if len(slopes) != len(x):
        raise ValueError("dydx_nodes must have the same length as x_nodes.")
    if not np.all(np.isfinite(slopes)):
        raise ValueError("dydx_nodes contains nan or inf.")

    numPoints = len(x)
    numDoubled = 2 * numPoints
    doubledX = np.repeat(x, 2)

    table = np.zeros((numDoubled, numDoubled))

    # f[zᵢ, ..., zᵢ₊ₖ] = ( f[zᵢ₊₁, ..., zᵢ₊ₖ] − f[zᵢ, ..., zᵢ₊ₖ₋₁] ) / ( zᵢ₊ₖ − zᵢ )

    table[:, 0] = np.repeat(y, 2)
    for row in range(1, numDoubled):
        if row % 2 == 1:
            table[row, 1] = slopes[row // 2]
        else:
            rise = table[row, 0] - table[row - 1, 0]
            run = doubledX[row] - doubledX[row - 1]
            table[row, 1] = rise / run
    for column in range(2, numDoubled):
        for row in range(column, numDoubled):
            change = table[row, column - 1] - table[row - 1, column - 1]
            distance = doubledX[row] - doubledX[row - column]
            table[row, column] = change / distance

    coefficients = np.diag(table).copy()
    return {"z": doubledX, "coeffs": coefficients, "table": table}


def hermite_interpolate(x_nodes, y_nodes, dydx_nodes, x_eval, return_derivative=False):
    """
    Evaluate H_ {2 n +1}( x_eval ) using nested ( Horner - like ) multiplication
    on the Newton form . If return_derivative = True , ALSO return
    H ’_ {2 n +1}( x_eval ) , computed analytically from the Newton form
    ( differentiate the nested product -- do NOT use a finite difference )


    Parameters
    ----------
    x_nodes : array_like
        Distinct nodes f(x_i) locations (any order).
    y_nodes : array_like
        Function values at x_nodes.
    dydx_nodes : array_like
        Derivative values at x_nodes.
    x_eval : float or 1-D ndarray
        Evaluation point(s).
    return_derivative : bool, default False
        Whether to include H'(x_eval) in the output.

    Returns
    -------
    dict
        Keys: ``"value"`` (matches ``x_eval`` shape), ``"derivative"``
        (matches ``x_eval`` shape or ``None``), ``"degree"`` (``2*n - 1``),
        and ``"extrapolating"`` (bool).

    Raises
    ------
    ValueError
        If inputs are empty, have mismatched lengths, contain duplicate nodes,
        or include NaN/Inf.
    TypeError
        If inputs are non-numeric.

    Example
    -------
    >>> r = hermite_interpolate([0, 1], [0, 1], [0, 0], 0.5, return_derivative=True)
    >>> float(r["value"]), float(r["derivative"]), r["degree"]
    (0.5, 1.5, 3)
    """

    result = hermite_coefficients(x_nodes, y_nodes, dydx_nodes)
    doubledX = result["z"]
    coefficients = result["coeffs"]

    x_eval = np.asarray(x_eval, dtype=float)

    extrapolating = bool(
        np.any(x_eval < doubledX.min()) or np.any(x_eval > doubledX.max())
    )
    if extrapolating:
        warnings.warn("Extrapolating: x_eval is outside the range of the nodes")

    last = len(coefficients) - 1
    value = coefficients[last] * np.ones_like(x_eval)
    derivative = np.zeros_like(x_eval)

    # work from the innermost bracket outward
    for i in range(last - 1, -1, -1):
        derivative = derivative * (x_eval - doubledX[i]) + value  # uses the OLD value
        value = value * (x_eval - doubledX[i]) + coefficients[i]

    return {
        "value": value,
        "derivative": derivative if return_derivative else None,
        "degree": last,
        "extrapolating": extrapolating,
    }


if __name__ == "__main__":
    f = np.exp  # f(x) = e^x
    fprime = np.exp  # f'(x) = e^x too
    grid = np.linspace(0, 1, 1001)
    x3 = np.array([0, 0.5, 1.0])
    x6 = np.linspace(0, 1, 6)

    # Test (i): is derivative information worth it?
    lag3 = lagrange_interpolate(x3, f(x3), grid)["value"]
    her3 = hermite_interpolate(x3, f(x3), fprime(x3), grid)["value"]
    lag6 = lagrange_interpolate(x6, f(x6), grid)["value"]

    at_q = 0.25
    print("(i) method                  | nodes | error at 0.25 | max error on grid")
    for name, nodes, fine, point in [
        (
            "Lagrange, 3 nodes (deg 2)",
            x3,
            lag3,
            lagrange_interpolate(x3, f(x3), at_q)["value"],
        ),
        (
            "Hermite,  3 nodes (deg 5)",
            x3,
            her3,
            hermite_interpolate(x3, f(x3), fprime(x3), at_q)["value"],
        ),
        (
            "Lagrange, 6 nodes (deg 5)",
            x6,
            lag6,
            lagrange_interpolate(x6, f(x6), at_q)["value"],
        ),
    ]:
        print(
            f"    {name} | {len(nodes):>5} | {abs(point - f(at_q)):.3e} "
            f"| {np.max(np.abs(fine - f(grid))):.3e}"
        )

    # Test (ii): interpolation conditions at every node
    r = hermite_interpolate(x3, f(x3), fprime(x3), x3, return_derivative=True)
    print("(ii) max |H(xi) - f(xi)|   =", np.max(np.abs(r["value"] - f(x3))))
    print("     max |H'(xi) - f'(xi)| =", np.max(np.abs(r["derivative"] - fprime(x3))))

    # Test (iii): exactness for x^5 - 3x^2 + 1, but not x^6
    def p(x):
        return x**5 - 3 * x**2 + 1

    def dp(x):
        return 5 * x**4 - 6 * x

    def q(x):
        return x**6

    def dq(x):
        return 6 * x**5

    Hp = hermite_interpolate(x3, p(x3), dp(x3), grid)["value"]
    Hq = hermite_interpolate(x3, q(x3), dq(x3), grid)["value"]
    print("(iii) max error for x^5 - 3x^2 + 1 =", np.max(np.abs(Hp - p(grid))))
    print("      max error for x^6             =", np.max(np.abs(Hq - q(grid))))

    # for x^6, f^(6)/6! = 1, so the error at x should equal prod (x - xi)^2 exactly
    xe = 0.25
    Hq_xe = hermite_interpolate(x3, q(x3), dq(x3), xe)["value"]
    print("      actual error at 0.25          =", q(xe) - Hq_xe)
    print("      prod (x - xi)^2 at 0.25       =", np.prod((xe - x3) ** 2))
