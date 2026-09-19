import math



def bisection(f, a, b, tol=1e-8, max_iter=100, error_type="absolute", func=None, verbose=False):
    """
    Find a root of f on [a, b] using the Bisection Method.

    Parameters
    ----------
    f : callable
        A function of one variable, f(x) -> float.
    a, b : float
        Endpoints of the initial bracketing interval, a < b.
    tol : float
        The tolerance for the chosen stopping criterion. Must be > 0.
    max_iter : int
        Maximum number of iterations allowed before giving up.
    error_type : str
        One of "absolute", "relative", "residual", "interval", or "all".
    verbose : bool
        If True, print a formatted iteration table as it runs.

    Returns
    -------
    dict
        Dictionary containing the root, iterations, convergence status,
        stopping reason, and iteration history (list of dictionaries).

    Raises
    ------
    TypeError
        If f is not callable.
    ValueError
    If a >= b.
    If tol <= 0.
    If max_iter is not a positive integer.
    If error_type is invalid.
    If f(a) and f(b) do not have opposite signs.

    func : callable ?? idk yet

    Example
    -------
    >>> result = bisection(lambda x: x**2 - 4, 0, 5)
    >>> result["root"]
    2.0
    """
    if not callable(f):
        raise TypeError("f must be a callable function.")
    if a >= b:
        raise ValueError("a must be less than b.")
    if tol <= 0:
        raise ValueError("tol must be greater than 0.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if error_type not in ["absolute", "relative", "residual", "interval", "all"]:
        raise ValueError("error_type must be one of 'absolute', 'relative', 'residual', 'interval', or 'all'.")
    fa = f(a)
    fb = f(b)
    a_priori_iterations = int(math.ceil(math.log2((b - a) / tol)))
    if fa == 0:
        return {"root": a, "iterations": 0, "converged": True, "reason": "f(a) is zero.", "history": [], "a_priori_iterations": a_priori_iterations}
    if fb == 0:
        return {"root": b, "iterations": 0, "converged": True, "reason": "f(b) is zero.", "history": [], "a_priori_iterations": a_priori_iterations}
    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")


    history = []
    prev_p = None

    for k in range(1, max_iter + 1):
        interval_width = b - a
        p_n = a + (b - a) / 2.0
        fp_n = f(p_n)

        if prev_p is not None:
            abs_error = abs(p_n - prev_p)
            rel_error = abs_error / abs(p_n)         
        else:
            abs_error = None
            rel_error = None

        residual = abs(fp_n)
        history.append({
            "n": k,
            "a": a,
            "b": b,
            "p": p_n,
            "f_p": fp_n,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "interval_width": interval_width,
        })

        if fp_n == 0.0:
            root = p_n
            converged = True
            stop_reason = "Exact root found."
            break

        stop = False
        if error_type == "absolute" and abs_error is not None and abs_error < tol:
            stop = True
            stop_reason = f"Absolute error {abs_error} < tol {tol}."
        elif error_type == "relative" and rel_error is not None and rel_error < tol:
            stop = True
            stop_reason = f"Relative error {rel_error} < tol {tol}."
        elif error_type == "residual" and residual < tol:
            stop = True
            stop_reason = f"Residual {residual} < tol {tol}."
        elif error_type == "interval" and interval_width < tol:
            stop = True
            stop_reason = f"Interval width {interval_width} < tol {tol}."
        elif error_type == "all":
            if (
                abs_error is not None
                and abs_error < tol
                and rel_error < tol
                and residual < tol
                and interval_width < tol
            ):
                stop = True
                stop_reason = f"All stopping criteria satisfied at tolerance = {tol}."

        if stop:
            root = p_n
            converged = True
            break

        if k == max_iter:
            root = p_n
            converged = False
            stop_reason = "Maximum iterations reached."
            break

        if fa * fp_n < 0:
            b = p_n
            fb = fp_n
        else:
            a = p_n
            fa = fp_n

        prev_p = p_n

    return {
        "root": root,
        "iterations": len(history),
        "converged": converged,
        "reason": stop_reason,
        "history": history,
        "a_priori_iterations": a_priori_iterations,
    }



if __name__ == "__main__":
    f = lambda x: x**2 - 4
    result = bisection(f, 0, 5)
    print(result)
    print(result["root"]) 

