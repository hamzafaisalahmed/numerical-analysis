"""Newton forward- and backward-difference interpolation (Problem 2)."""

import warnings
import numpy as np


def difference_table(y_nodes):
    """Build the full forward-difference table.

    Parameters
    ----------
    y_nodes : array_like of float
        Function values y_i at the nodes.

    Returns
    -------
    list of 1-D ndarray
        table[0] = y, table[1] = Delta y, table[2] = Delta^2 y, ...,
        table[n-1] (length 1) for n nodes. An empty input gives [].
        The same table also contains all backward differences:
        nabla^k y_i = Delta^k y_{i-k} = table[k][i-k].

    Raises
    ------
    ValueError
        If y_nodes cannot be converted to a 1-D float array.

    Example
    -------
    >>> t = difference_table([1, 4, 9, 16])
    >>> [row.tolist() for row in t]
    [[1.0, 4.0, 9.0, 16.0], [3.0, 5.0, 7.0], [2.0, 2.0], [0.0]]
    """
    y = np.asarray(y_nodes, dtype=float)
    if y.ndim != 1:
        raise ValueError("y_nodes must be one-dimensional.")
    n = len(y)
    if n == 0:
        return []

    table = [y.copy()]  # table[0] is y itself
    for i in range(1, n):
        previous = table[i - 1]
        table.append(previous[1:] - previous[:-1])
    return table


def print_difference_table(x_nodes, y_nodes, decimals=6):
    """Pretty-print the difference table in triangular textbook layout.

    Parameters
    ----------
    x_nodes : array_like of float
        Node abscissas.
    y_nodes : array_like of float
        Function values at the nodes (same length as x_nodes).
    decimals : int, optional
        Digits after the decimal point (default 6).

    Returns
    -------
    None
        The table is printed to standard output.

    Raises
    ------
    ValueError
        If x_nodes and y_nodes have different lengths.

    Example
    -------
    >>> print_difference_table([0, 1, 2], [1, 4, 9], decimals=1)   # doctest: +SKIP
               x        f(x)          Δy       Δ^2y
    ------------------------------------------------
             0.0         1.0         3.0         2.0
             1.0         4.0         5.0
             2.0         9.0
    """
    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have the same length.")

    n = len(x_nodes)
    if n == 0:
        print("(empty table)")
        return
    table = difference_table(y_nodes)

    headers = ["x", "f(x)"] + ["Δy" if k == 1 else f"Δ^{k}y" for k in range(1, n)]

    grid = np.full((n, n + 1), np.nan)
    for i in range(n):
        grid[i, 0] = x_nodes[i]
        for k in range(n - i):
            grid[i, k + 1] = table[k][i]

    col_width = max(12, decimals + 6)

    header_row = "".join(f"{h:>{col_width}}" for h in headers)
    print(header_row)
    print("-" * len(header_row))

    for i in range(n):
        row_str = ""
        for j in range(n + 1):
            val = grid[i, j]
            if np.isnan(val):
                row_str += f"{'':>{col_width}}"
            else:
                row_str += f"{val:>{col_width}.{decimals}f}"
        print(row_str)


def check_equal_spacing(x_nodes, rtol=1e-9):
    """Check that the nodes are equally spaced and return the step h.

    Parameters
    ----------
    x_nodes : array_like of float
        Node abscissas (any order; they are sorted internally).
    rtol : float, optional
        Relative tolerance on the deviation of each spacing from h
        (default 1e-9).

    Returns
    -------
    float
        The common spacing h = (max - min) / (n - 1).

    Raises
    ------
    ValueError
        If fewer than 2 nodes are given, nodes are not distinct, or the
        spacing is not uniform (the message reports the largest deviation).
    TypeError
        If x_nodes is not numeric.

    Example
    -------
    >>> round(check_equal_spacing([0, 0.2, 0.4, 0.6]), 10)
    0.2
    """
    try:
        x = np.sort(np.asarray(x_nodes, dtype=float))
    except (TypeError, ValueError):
        raise TypeError("x_nodes must be numeric.")
    if len(x) < 2:
        raise ValueError("Not enough data: at least 2 nodes are required.")

    steps = np.diff(x)
    h = (x[-1] - x[0]) / (len(x) - 1)
    if h <= 0:
        raise ValueError("Nodes must be distinct.")
    worst = float(np.max(np.abs(steps - h)))
    if worst > rtol * h:
        raise ValueError(
            f"Nodes are not equally spaced. Largest spacing deviation is {worst:g} "
            f"(expected h ~ {h:g}). Use Lagrange interpolation instead."
        )
    return float(h)


def _prepare(x_nodes, y_nodes, degree):
    """Validate inputs, sort nodes (with y), return (x, y, table, h, n, deg)."""
    try:
        x = np.asarray(x_nodes, dtype=float)
        y = np.asarray(y_nodes, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("x_nodes and y_nodes must be numeric.")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be one-dimensional.")
    if len(x) != len(y):
        raise ValueError(
            f"Length mismatch: x_nodes has {len(x)} entries, y_nodes has {len(y)}."
        )

    # Sort nodes and y together so unsorted/descending data works.
    order = np.argsort(x)
    x, y = x[order], y[order]

    h = check_equal_spacing(x)  # also catches < 2 nodes and unequal spacing
    n = len(x)

    if degree is None:
        deg = n - 1
    else:
        if isinstance(degree, bool) or int(degree) != degree:
            raise ValueError("degree must be an integer.")
        deg = int(degree)
        if deg < 1:
            raise ValueError(f"Given degree {deg} is too small (must be >= 1).")
        if deg > n - 1:
            raise ValueError(
                f"Given degree {deg} is too large (at most {n - 1} for {n} nodes)."
            )

    table = difference_table(y)  # computed once, reused by every evaluation
    return x, y, table, h, n, deg


def _warn_if_extrapolating(x, x_eval):
    """Issue a warning if any x_eval lies outside [min(x), max(x)]."""
    xe = np.asarray(x_eval, dtype=float)
    if np.any(xe < x[0]) or np.any(xe > x[-1]):
        warnings.warn(
            f"Extrapolating: x_eval lies outside the node range [{x[0]:g}, {x[-1]:g}].",
            stacklevel=3,
        )


def _forward_scalar(x, table, h, n, deg, degree_given, xe):
    """Forward formula at one point, using a prepared (sorted) table."""
    k = 0
    if degree_given:
        for i in range(n):
            if xe >= x[i]:
                k = i
            else:
                break
        if k + deg >= n:
            k = n - 1 - deg

    s = (xe - x[k]) / h
    term_0 = table[0][k]
    terms = [term_0]
    value = term_0

    s_prod = 1.0
    factorial = 1
    for j in range(1, deg + 1):
        s_prod *= s - (j - 1)
        factorial *= j
        term_j = (s_prod / factorial) * table[j][k]
        terms.append(term_j)
        value += term_j

    error_estimate = None
    nl = deg + 1
    if k + nl < n:
        error_estimate = abs(((s_prod * (s - deg)) / (factorial * nl)) * table[nl][k])

    return {
        "value": value,
        "degree": deg,
        "base_node": x[k],
        "s": s,
        "terms": terms,
        "error_estimate": error_estimate,
    }


def _backward_scalar(x, table, h, n, deg, degree_given, xe):
    """Backward formula at one point, using a prepared (sorted) table."""
    k = n - 1
    if degree_given:
        for i in range(n - 1, -1, -1):
            if xe <= x[i]:
                k = i
            else:
                break
        if k - deg < 0:
            k = deg

    t = (xe - x[k]) / h
    term_0 = table[0][k]
    terms = [term_0]
    value = term_0

    t_prod = 1.0
    factorial = 1
    for j in range(1, deg + 1):
        t_prod *= t + (j - 1)
        factorial *= j
        term_j = (t_prod / factorial) * table[j][k - j]  # nabla^j y_k = Delta^j y_{k-j}
        terms.append(term_j)
        value += term_j

    error_estimate = None
    nl = deg + 1
    if k - nl >= 0:
        error_estimate = abs(((t_prod * (t + deg)) / (factorial * nl)) * table[nl][k - nl])

    return {
        "value": value,
        "degree": deg,
        "base_node": x[k],
        "t": t,
        "terms": terms,
        "error_estimate": error_estimate,
    }


def newton_forward(x_nodes, y_nodes, x_eval, degree=None):
    """Newton forward-difference interpolation.

    If degree is None, all nodes are used (base node = smallest node).
    Otherwise degree+1 consecutive nodes are used, starting from the base
    node x_k = the largest node <= x_eval for which degree+1 nodes
    x_k, ..., x_{k+degree} still exist (so the formula is used where it
    is most accurate). Nodes may be given in any order; they are sorted
    together with y_nodes.

    Parameters
    ----------
    x_nodes : array_like of float
        Equally spaced node abscissas (any order).
    y_nodes : array_like of float
        Function values at the nodes (same length as x_nodes).
    x_eval : float or 1-D ndarray
        Point(s) at which to interpolate.
    degree : int, optional
        Polynomial degree, 1 <= degree <= n-1. None means n-1 (all nodes).

    Returns
    -------
    dict (scalar x_eval) with keys:
        "value"          -- interpolated value
        "degree"         -- degree used
        "base_node"      -- x_k used as the base node
        "s"              -- (x_eval - x_k) / h
        "terms"          -- list of the individual terms added
        "error_estimate" -- magnitude of the first OMITTED term
                            (None if no higher difference exists)
    ndarray (array x_eval)
        Only the interpolated values, one per entry of x_eval.

    Raises
    ------
    ValueError
        Unequally spaced or non-distinct nodes, degree not an integer or
        out of range, mismatched lengths, fewer than 2 nodes.
    TypeError
        Non-numeric input.

    Warns
    -----
    UserWarning
        If x_eval lies outside [min(x_nodes), max(x_nodes)] (extrapolation).

    Example
    -------
    >>> x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    >>> y = np.exp(x)
    >>> r = newton_forward(x, y, 0.05, degree=3)
    >>> round(float(r["value"]), 4), float(r["base_node"])
    (1.0513, 0.0)
    """
    x, y, table, h, n, deg = _prepare(x_nodes, y_nodes, degree)
    _warn_if_extrapolating(x, x_eval)
    given = degree is not None

    if np.ndim(x_eval) > 0:
        return np.array([
            _forward_scalar(x, table, h, n, deg, given, float(xe))["value"]
            for xe in np.asarray(x_eval, dtype=float).ravel()
        ]).reshape(np.shape(x_eval))
    return _forward_scalar(x, table, h, n, deg, given, float(x_eval))


def newton_backward(x_nodes, y_nodes, x_eval, degree=None):
    """Newton backward-difference interpolation.

    If degree is None, all nodes are used (base node = largest node).
    Otherwise degree+1 consecutive nodes are used, ending at the base node
    x_k = the smallest node >= x_eval for which degree+1 nodes
    x_{k-degree}, ..., x_k exist. Nodes may be given in any order; they
    are sorted together with y_nodes.

    Parameters
    ----------
    x_nodes : array_like of float
        Equally spaced node abscissas (any order).
    y_nodes : array_like of float
        Function values at the nodes (same length as x_nodes).
    x_eval : float or 1-D ndarray
        Point(s) at which to interpolate.
    degree : int, optional
        Polynomial degree, 1 <= degree <= n-1. None means n-1 (all nodes).

    Returns
    -------
    dict (scalar x_eval) with keys:
        "value"          -- interpolated value
        "degree"         -- degree used
        "base_node"      -- x_k used as the base node
        "t"              -- (x_eval - x_k) / h
        "terms"          -- list of the individual terms added
        "error_estimate" -- magnitude of the first OMITTED term
                            (None if no higher difference exists)
    ndarray (array x_eval)
        Only the interpolated values, one per entry of x_eval.

    Raises
    ------
    ValueError
        Unequally spaced or non-distinct nodes, degree not an integer or
        out of range, mismatched lengths, fewer than 2 nodes.
    TypeError
        Non-numeric input.

    Warns
    -----
    UserWarning
        If x_eval lies outside [min(x_nodes), max(x_nodes)] (extrapolation).

    Example
    -------
    >>> x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    >>> y = np.exp(x)
    >>> r = newton_backward(x, y, 0.95, degree=3)
    >>> round(float(r["value"]), 4), float(r["base_node"])
    (2.5858, 1.0)
    """
    x, y, table, h, n, deg = _prepare(x_nodes, y_nodes, degree)
    _warn_if_extrapolating(x, x_eval)
    given = degree is not None

    if np.ndim(x_eval) > 0:
        return np.array([
            _backward_scalar(x, table, h, n, deg, given, float(xe))["value"]
            for xe in np.asarray(x_eval, dtype=float).ravel()
        ]).reshape(np.shape(x_eval))
    return _backward_scalar(x, table, h, n, deg, given, float(x_eval))


if __name__ == "__main__":
    x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y = np.exp(x)

    print("=== Difference table of e^x ===")
    print_difference_table(x, y, decimals=6)

    print("\n=== Test (i): accuracy vs degree ===")
    for name, fn, xe in (("forward ", newton_forward, 0.05),
                         ("backward", newton_backward, 0.95)):
        print(f"\nNewton {name}, e^{xe}  (true = {np.exp(xe):.8f})")
        print(f"{'deg':>4}{'value':>14}{'true error':>14}{'error_est':>14}{'base':>7}")
        for d in (1, 2, 3, 4):
            r = fn(x, y, xe, degree=d)
            est = "None" if r["error_estimate"] is None else f"{r['error_estimate']:.3e}"
            print(f"{d:>4}{r['value']:>14.8f}{abs(r['value'] - np.exp(xe)):>14.3e}"
                  f"{est:>14}{r['base_node']:>7.2f}")

    print("\n=== Test (ii): same polynomial, two formulas ===")
    xs = np.linspace(0, 1, 200)
    f_all = newton_forward(x, y, xs)
    b_all = newton_backward(x, y, xs)
    print(f"all 6 nodes : max|forward - backward| = {np.max(np.abs(f_all - b_all)):.3e}")
    f2 = newton_forward(x, y, xs, degree=2)
    b2 = newton_backward(x, y, xs, degree=2)
    print(f"degree = 2  : max|forward - backward| = {np.max(np.abs(f2 - b2)):.3e}")
    print(f"   max true err forward  = {np.max(np.abs(f2 - np.exp(xs))):.3e}")
    print(f"   max true err backward = {np.max(np.abs(b2 - np.exp(xs))):.3e}")

    print("\n=== Unsorted nodes (descending) ===")
    xd, yd = x[::-1], y[::-1]
    print("forward  matches sorted:", np.isclose(newton_forward(xd, yd, 0.05, 3)["value"],
                                                 newton_forward(x, y, 0.05, 3)["value"]))
    print("backward matches sorted:", np.isclose(newton_backward(xd, yd, 0.95, 3)["value"],
                                                 newton_backward(x, y, 0.95, 3)["value"]))

    print("\n=== Error handling ===")
    for label, call in [
        ("unequal spacing", lambda: newton_forward([0, 1, 2.5, 3], [0, 1, 2, 3], 1)),
        ("degree too big ", lambda: newton_forward(x, y, 0.3, degree=6)),
        ("degree too small", lambda: newton_forward(x, y, 0.3, degree=0)),
        ("length mismatch", lambda: newton_backward([0, 1, 2], [0, 1], 1)),
        ("one node       ", lambda: newton_forward([0], [1], 0)),
    ]:
        try:
            call()
        except ValueError as e:
            print(f"{label}: ValueError -> {e}")

    print("\n=== Extrapolation warning ===")
    for fn in (newton_forward, newton_backward):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn(x, y, 1.2, degree=2)
            print(f"{fn.__name__}: warning raised: {bool(w)}")