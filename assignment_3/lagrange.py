import math
import warnings
import numpy as np


def check_data(x_nodes, y_nodes=None):
    """
    Check the nodes (and values) and convert them to numpy arrays.

    Parameters
    ----------
    x_nodes : list or array of nodes
    y_nodes : list or array of values (optional)

    Returns
    -------
    x_nodes, y_nodes : numpy arrays (y_nodes is None if not given)

    Raises
    ------
    TypeError  -- input is not numeric
    ValueError -- no nodes, different lengths, nan/inf, repeated nodes

    Example
    -------
    >>> x, y = check_data([0, 1], [2, 3])
    """
    try:
        x_nodes = np.asarray(x_nodes, dtype=float)
        if y_nodes is not None:
            y_nodes = np.asarray(y_nodes, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("input must be numeric.")

    if len(x_nodes) < 1:
        raise ValueError("need at least 1 node.")
    if y_nodes is not None and len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have the same length.")
    if not np.all(np.isfinite(x_nodes)):
        raise ValueError("x_nodes contains nan or inf.")
    if y_nodes is not None and not np.all(np.isfinite(y_nodes)):
        raise ValueError("y_nodes contains nan or inf.")

    for i in range(len(x_nodes)):
        for j in range(i + 1, len(x_nodes)):
            if abs(x_nodes[i] - x_nodes[j]) <= 1e-12 * max(abs(x_nodes[i]), abs(x_nodes[j])):
                raise ValueError("repeated node: " + str(x_nodes[i]))
    return x_nodes, y_nodes


def lagrange_basis(x_nodes, k, x_eval):
    """
    Evaluate the k-th Lagrange basis polynomial L_{n,k} at x_eval.

    Parameters
    ----------
    x_nodes : n+1 distinct nodes
    k : index of the basis polynomial, 0 <= k <= n
    x_eval : float or 1-D numpy array

    Returns
    -------
    float or numpy array (same shape as x_eval)

    Raises
    ------
    ValueError -- k out of range, or nodes not distinct

    Example
    -------
    >>> float(lagrange_basis([0, 1, 2], 1, 1.0))
    1.0
    """
    x_nodes, _ = check_data(x_nodes)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x_nodes)
    if k < 0 or k >= n:
        raise ValueError("k is out of range.")

    L = np.ones_like(x_eval, dtype=float)
    for i in range(n):
        if i != k:
            L *= (x_eval - x_nodes[i]) / (x_nodes[k] - x_nodes[i])
    return L


def lagrange_interpolate(x_nodes, y_nodes, x_eval, return_basis=False):
    """
    Evaluate the Lagrange interpolating polynomial at x_eval.

    Parameters
    ----------
    x_nodes : distinct nodes (any order)
    y_nodes : values at the nodes
    x_eval : float or 1-D numpy array
    return_basis : if True, also return the basis values

    Returns
    -------
    dict with keys "value", "degree", "basis" (list or None),
    "extrapolating" (True if x_eval is outside the range of the nodes)

    Raises
    ------
    ValueError -- different lengths, no nodes, repeated nodes, nan/inf
    TypeError  -- input is not numeric

    Example
    -------
    >>> float(lagrange_interpolate([0, 1, 2], [0, 1, 4], 1.5)["value"])
    2.25
    """
    x_nodes, y_nodes = check_data(x_nodes, y_nodes)
    x_eval = np.asarray(x_eval, dtype=float)

    n = len(x_nodes)
    extrapolating = bool(np.any(x_eval < x_nodes.min()) or np.any(x_eval > x_nodes.max()))
    if extrapolating:
        warnings.warn("Extrapolating: x_eval is outside the range of the nodes")

    val = np.zeros_like(x_eval, dtype=float)
    basis_list = []
    for k in range(n):
        L_k = lagrange_basis(x_nodes, k, x_eval)
        val += y_nodes[k] * L_k
        if return_basis:
            basis_list.append(L_k)

    return {
        "value": val,
        "degree": n - 1,
        "basis": basis_list if return_basis else None,
        "extrapolating": extrapolating
    }


def interpolation_error_bound(x_nodes, x_eval, deriv_bound):
    """
    Theoretical error bound M/(n+1)! * |prod_i (x_eval - x_i)|.

    Parameters
    ----------
    x_nodes : n+1 distinct nodes
    x_eval : float or 1-D numpy array
    deriv_bound : M, an upper bound for |f^(n+1)| on the interval

    Returns
    -------
    float or numpy array (same shape as x_eval)

    Raises
    ------
    ValueError -- repeated nodes, no nodes, nan/inf
    TypeError  -- input is not numeric

    Example
    -------
    >>> float(interpolation_error_bound([0, 1], 0.5, 2.0))
    0.25
    """
    x_nodes, _ = check_data(x_nodes)
    x_eval = np.asarray(x_eval, dtype=float)

    n_plus_1_factorial = math.factorial(len(x_nodes))
    prod_term = np.ones_like(x_eval, dtype=float)
    for xi in x_nodes:
        prod_term *= np.abs(x_eval - xi)

    return (deriv_bound / n_plus_1_factorial) * prod_term


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # (i) Kronecker property
    x = np.array([0, 1.5, 2, 4.2])
    worst = 0
    for k in range(4):
        for j in range(4):
            expected = 0
            if k == j:
                expected = 1
            worst = max(worst, abs(lagrange_basis(x, k, x[j]) - expected))
    print("(i) max |L_k(x_j) - delta_kj| =", worst)

    grid = np.linspace(-3, 7, 101)
    total = np.zeros_like(grid)
    for k in range(4):
        total += lagrange_basis(x, k, grid)
    print("    max |sum L_k(x) - 1|      =", np.max(np.abs(total - 1)))

    # (ii) Exactness for f(x) = 2x^3 - x + 5
    x = np.array([-1.3, 0.2, 1.1, 2.7])
    y = 2 * x**3 - x + 5
    grid = np.linspace(-1.3, 2.7, 2001)
    exact = 2 * grid**3 - grid + 5
    P = lagrange_interpolate(x, y, grid)["value"]
    print("(ii) max error =", np.max(np.abs(exact - P)))

    # (iii) Runge phenomenon
    grid = np.linspace(-1, 1, 2001)
    f_grid = 1 / (1 + 25 * grid**2)
    print("(iii) n | equispaced | Chebyshev")
    for n in [5, 10, 15, 20]:
        x_eq = np.linspace(-1, 1, n + 1)
        i = np.arange(n + 1)
        x_ch = np.cos((2 * i + 1) * np.pi / (2 * (n + 1)))
        y_eq = 1 / (1 + 25 * x_eq**2)
        y_ch = 1 / (1 + 25 * x_ch**2)
        P_eq = lagrange_interpolate(x_eq, y_eq, grid)["value"]
        P_ch = lagrange_interpolate(x_ch, y_ch, grid)["value"]
        print(n, "|", np.max(np.abs(f_grid - P_eq)), "|", np.max(np.abs(f_grid - P_ch)))

        if n == 10:
            plt.plot(grid, f_grid, "k", label="f(x)")
            plt.plot(grid, P_eq, "orange", label="Equispaced, n=10")
            plt.plot(grid, P_ch, "brown", label="Chebyshev, n=10")
            plt.plot(x_eq, y_eq, "o", color="purple", label="Equispaced nodes")
            plt.plot(x_ch, y_ch, "s", color="green", label="Chebyshev nodes")
            plt.xlabel("x")
            plt.ylabel("f(x), P(x)")
            plt.legend()
            plt.savefig("runge.png")