import numpy as np

from fixed_point import estimate_order


def newton(
    f,
    p0,
    fprime=None,
    tol=1e-10,
    max_iter=100,
    numeric_derivative=False,
    h=1e-6,
    verbose=False,
):
    """
    Find a root of f using Newton's Method, p_{n+1} = p_n - f(p_n)/f'(p_n).

    If fprime is None or numeric_derivative=True, approximate f'(p_n)
    using the central-difference formula
        f'(p_n) ~= (f(p_n+h) - f(p_n-h)) / (2h)

    Parameters
    ----------
    f : callable
        The function for which we are trying to find a root.
    p0 : int, float
        Initial guess for a root of f.
    fprime : callable, optional
        The derivative of f. If not provided, a numerical derivative will be calculated.
    tol : float, optional
        The stopping criteria: stops when difference between two consecutive iterations is less than tol.
    max_iter : int, optional
        The maximum number of iterations to perform.
    numeric_derivative : bool, optional
        If True, calculates and uses a numerical derivative for fprime, disregarding fprime if provided.
    h : float, optional
        Used in central difference formula to approximate derivative
    verbose : bool, optional
        If True, prints the details of each iteration.
    Returns
        -------
        dict with keys:
            "root", "iterations", "converged", "reason", "history"
            (list of dicts: {n, p, f_p, fprime_p, diff}),
            "order_estimate" -- via estimate_order() from Problem 1,
                                reused here (imported, did not duplicate it)
    Raises
        ------
        ValueError -- tol <= 0, max_iter <= 0
        TypeError -- f is not callable (or fprime, if supplied, is not)
        ZeroDivisionError
            -- raised (not silently caught) if |f'(p_n)| falls below a
               small threshold at any iteration, since Newton's Method is
               not defined there; the message must report which iteration
               and the value of f'(p_n).
    """
    MIN = 1e-12
    MAX_CONSECUTIVE_CYCLES = 10
    if max_iter <= 0:
        raise ValueError("max iterations must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if not (callable(f)):
        raise TypeError("f must be a callable function")
    if fprime is not None and not callable(fprime):
        raise TypeError("fprime if provided must be a callable function")
    if fprime is None or numeric_derivative:
        fprime = lambda p_n: (f(p_n + h) - f(p_n - h)) / (2 * h)

    n = 0
    p = p0
    converged = False
    diff = 0
    history = []
    cycle = 0
    reason = None
    history.append({"n": n, "p": p, "f_p": None, "fprime_p": None, "diff": None})
    if verbose:
        print(f"n={n:3d}  p_n={p:.12f}  f(p_n-1)={None}  f'(p_n-1)={None}  diff={None}")
    while n < max_iter:
        n += 1
        f_p = f(p)
        fprime_p = fprime(p)
        if abs(fprime_p) < MIN:
            raise ZeroDivisionError(
                f"at iteration: {n}, the value of f'(p_n) becomes {fprime_p}"
            )
        p_new = p - f_p / fprime_p
        diff = abs(p_new - p)
        p = p_new

        history.append({"n": n, "p": p, "f_p": f_p, "fprime_p": fprime_p, "diff": diff})

        if verbose:
            print(
                f"n={n:3d}  p_n={p:.12f}  f(p_n-1)={f_p:.3e}  f'(p_n-1)={fprime_p:.3e}  diff={diff:.3e}"
            )
        if diff < tol:
            converged = True
            reason = "|p_n+1 - p_n| < tol"
            break
        if n >= 2:
            if abs(p - history[n - 2]["p"]) < MIN:
                cycle += 1
            else:
                cycle = 0
        if cycle > MAX_CONSECUTIVE_CYCLES:
            reason = "cycle detected"
            break
    if not converged:
        p = None
        if n == max_iter and reason is None:
            reason = f"Reached iteration {max_iter} without converging"
    sequence_of_iterations = []
    for i in range(len(history)):
        sequence_of_iterations.append(history[i]["p"])
    order = estimate_order(sequence_of_iterations)
    if verbose:
        print(
            f"Converged: {converged}\nReason: {reason}\nRoot: {p}\nIterations: {n}\nOrder of convergence: {order['order']}"
        )
    return {
        "root": p,
        "iterations": n,
        "converged": converged,
        "reason": reason,
        "history": history,
        "order_estimate": order,
    }


if __name__ == "__main__":
    print("Test i\n")
    p0 = 1
    f = lambda x: x**2 - 2
    result_analytic = newton(f, p0, lambda x: 2 * x)
    result_numeric = newton(f, p0, numeric_derivative=True)
    print(
        f"===Convergence===\nAnalytic: {result_analytic['converged']}\tNumeric: {result_numeric['converged']}"
    )
    print(
        f"\n===Root===\nAnalytic: {result_analytic['root']}\tNumeric: {result_numeric['root']}"
    )
    print(
        f"\n===Order of convergence===\nAnalytic: {result_analytic['order_estimate']['order']}\tNumeric: {result_numeric['order_estimate']['order']}"
    )
    print(
        f"\n===Number of iterations===\nAnalytic: {result_analytic['iterations']}\tNumeric: {result_numeric['iterations']}"
    )
    print(
        f"\n===Accuracy===\nAnalytic: {abs(result_analytic['root'] - np.sqrt(2)):.3e}\tNumeric: {abs(result_numeric['root'] - np.sqrt(2)):.3e}"
    )

    print("\nTest ii\n")
    p0 = 0.5
    result = newton(f, p0, numeric_derivative=True)
    print(f"Convergence: {result['converged']}")
    print(f"Root: {result['root']}")
    print(f"Order of convergence: {result['order_estimate']['order']}")

    print("\nTest iii\n")
    f = lambda x: x**3 - 2 * x + 2
    p0 = 0
    result = newton(f, p0, numeric_derivative=True)
    print(f"Using p0 = {p0}")
    print(f"Convergence: {result['converged']}")
    print(f"Root: {result['root']}")
    print(f"Reason: {result['reason']}")

    print("\nIf we used a different p0 such as p0 = 2: ")
    p0 = 2
    result = newton(f, p0, numeric_derivative=True)
    print(f"Using p0 = {p0}")
    print(f"Convergence: {result['converged']}")
    print(f"Root: {result['root']}")
    print(f"Reason: {result['reason']}")
