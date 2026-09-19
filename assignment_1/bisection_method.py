import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# helper functions for the different error checking techniques

def interval(a , b):
     return b-a


def Residual(f , pn):
     return abs(f(pn))

def Absolute_error(p_curr  , p_prev):
    if p_prev is None:
          return None

    return abs(p_curr - p_prev)

def Relative_error(p_curr  , p_prev , eps=1e-12):
    if p_prev is None:
        return None

    denom  = abs(p_curr)
    if denom <= eps:
        return abs(p_curr - p_prev)
    else:
         return (abs(p_curr - p_prev) / denom)


# Another helper function to bring all the error chekcing together 

def ErorChecker(error_type , tol, interval_w, residual, abs_err, rel_err):
    """
    Evaluates whether the active error_type criterion is satisfied.
    Returns (converged: bool, reason: str).
    """

    has_prev = abs_err is not None  #for iteration 1 there is not  previous and absolute and relative error are none

    if error_type == "interval":
        if interval_w < tol:
            return True, f"Interval width ({interval_w:.2e}) < tol ({tol:.2e})"

    if error_type == "residual":
        if residual < tol:
            return True, f"Residual |f(p)| ({residual:.2e}) < tol ({tol:.2e})"

    if  error_type == "absolute":
        if has_prev and abs_err < tol:
            return True, f"Absolute error ({abs_err:.2e}) < tol ({tol:.2e})"

    if  error_type == "relative":
        if has_prev and rel_err < tol:
            return True, f"Relative error ({rel_err:.2e}) < tol ({tol:.2e})"

    if error_type == "all":
        if has_prev and rel_err < tol and abs_err < tol and residual < tol and interval_w < tol :
           return True, "All stopping criteria satisfied simultaneously."

    return False , "";


def bisection (f, a, b, tol=1e-8, max_iter=100,  error_type="absolute", func=None, verbose=False):
    """
    Find a root of f on [a, b] using the Bisection Method.

    Parameters
    ----------
    f : callable
        A function of one variable, f(x) -> float.
    a, b : float
        Endpoints of the initial bracketing interval, a < b.
    tol : float, optional
        The tolerance for the chosen stopping criterion (default: 1e-8).
    max_iter : int, optional
        Maximum number of iterations allowed before giving up (default: 100).
    error_type : str, optional
        Stopping criterion: 'absolute', 'relative', 'residual', 'interval', or 'all'.
    func : None, optional
        Unused compatibility parameter.
    verbose : bool, optional
        If True, prints a formatted iteration table.

    Returns
    -------
    dict
        Dictionary containing root, iterations, converged, reason, history, and a_priori_iterations.

    Raises
    ------
    TypeError
        If f is not callable.
    ValueError
        If a >= b, tol <= 0, max_iter <= 0, invalid error_type, or f(a)*f(b) >= 0.

    Example
    -------
    >>> res = bisection(lambda x: x**2 - 2, 1.0, 2.0, tol=1e-4)
    >>> round(res["root"], 4)
    1.4142
    """


    if not callable(f): 
        raise TypeError(
            f"Parameter 'f' must be a callable function (e.g., lambda x: ...). Got type {type(f).__name__}."
        ) 

    if a >= b :
        raise ValueError( a  ,  " must be strictly less than",   b)

    if tol <=0: 
        raise ValueError ("Tolerance has to be greater than 0")

    if max_iter <= 0:
        raise ValueError ("Number of iteration cant be 0 or less than 0")

    allowed_errors = {"absolute", "relative", "residual", "interval", "all"}
    if error_type  not in allowed_errors:
        raise ValueError ("kindly enter a acceptable error type (absolute, relative, residual, interval, all) ") 


    fa = f(a)
    fb = f(b)

    if fa == 0:
        return {
            "root": float(a), "iterations": 0, "converged": True,
            "reason": "Exact root found at left boundary a.", "history": [],
            "a_priori_iterations": int(np.ceil(np.log2((b - a) / tol)))
        }

    if fb == 0:
            return {
                "root": float(b), "iterations": 0, "converged": True,
                "reason": "Exact root found at right boundary b.", "history": [],
                "a_priori_iterations": int(np.ceil(np.log2((b - a) / tol)))
            }


    if np.sign(fa) * np.sign(fb) >= 0:
         raise ValueError ("root is not bracketed product of fa and fb must have opp signs")


    p_prev = None 
    History = []
    converged  = False
    Reason = f"Maximum iterations ({max_iter}) reached without satisfying stopping criterion '{error_type}'."

    a_priori_iterations = int(np.ceil(np.log2((b-a) / tol)))

    for n in range (1 , max_iter+1):
        p = a + (b-a)/2 
        fp = f(p)

        

        
        interval_w = interval(a, b)
        residual = Residual(f, p)
        abs_err = Absolute_error(p, p_prev)
        rel_err = Relative_error(p, p_prev)


        History.append({
                    "n": n,
                    "a": a,
                    "b": b,
                    "p": p,
                    "f_p": fp,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "interval_width": interval_w,
                })

        if verbose:
            print(f"{n:<5} | {a:<14.8f} | {b:<14.8f} | {p:<14.8f} | {fp:<14.6e} | {interval_w:<12.6e}")
        
        result , stop_reason = ErorChecker(error_type, tol , interval_w , residual , abs_err , rel_err)

        if result:
            converged = True
            Reason = stop_reason
            break

        if fp == 0:
            converged = True
            Reason = f"Exact root hit at midpoint p = {p}"
            break

        if np.sign(fa) * np.sign(fp) < 0:
            b = p
            fb = fp
        else:
            a = p
            fa = fp

        p_prev = p 

    return {
    "root": float(p),
    "iterations": len(History),
    "converged": converged,
    "reason": Reason,
    "history": History,
    "a_priori_iterations": a_priori_iterations,
    }


def plot_convergence(result):
    history = result.get("history", [])

    # edge case
    if not history:
        raise ValueError("Cannot plot convergence: history list is empty.")

    iterations = np.array([row["n"] for row in history])
    interval_widths = np.array([row["interval_width"] for row in history])

    initial_width = interval_widths[0]
    theoretical_widths = initial_width * (0.5 ** (iterations - 1))

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        iterations,
        interval_widths,
        "bo-",
        label="Empirical Interval Width (b_n - a_n)",
        linewidth=1.5,
        markersize=5,
    )
    plt.semilogy(
        iterations,
        theoretical_widths,
        "r--",
        label="Theoretical Rate: W_1 \\cdot (1/2)^{n-1}",
        linewidth=2,
    )

    plt.xlabel("Iteration Number (n)")
    plt.ylabel("Interval Width (log scale)")
    plt.title(
        f"Bisection Method Convergence Rate (Total Iterations: {len(iterations)})"
    )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_filename = "bisection_convergence.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()  # Close figure to free memory

    print(f"Convergence plot saved successfully to {output_filename}")


if __name__ == "__main__":
    print("=" * 80)
    print("NUMERICAL ANALYSIS ASSIGNMENT 1 - PROBLEM 2 TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test Case (i): f(x) = x^3 - x - 2 on [1, 2], error_type="interval", tol=1e-6
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30 + " TEST CASE (i) " + "=" * 30)
    f1 = lambda x: x**3 - x - 2
    res1 = bisection(f1, 1.0, 2.0, tol=1e-6, error_type="interval", verbose=False)

    print(f"Target function: f(x) = x^3 - x - 2 on [1.0, 2.0]")
    print(f"Root: {res1['root']}")
    print(f"Converged: {res1['converged']}")
    print(f"Reason: {res1['reason']}")
    print(f"Actual Iterations: {res1['iterations']}")
    print(f"A Priori Iterations: {res1['a_priori_iterations']}")

    # Print first 3 and last 3 rows of history as mandated
    print("\nHistory (First 3 and Last 3 Iterations):")
    h = res1["history"]
    sample_history = h[:3] + h[-3:] if len(h) >= 6 else h
    print(f"{'n':<4} | {'a':<12} | {'b':<12} | {'p':<12} | {'f(p)':<14} | {'Width':<12}")
    print("-" * 75)
    for row in sample_history:
        print(f"{row['n']:<4} | {row['a']:<12.6f} | {row['b']:<12.6f} | {row['p']:<12.6f} | {row['f_p']:<14.6e} | {row['interval_width']:<12.6e}")
        if row['n'] == 3 and len(h) >= 6:
            print(" ...  |     ...      |     ...      |     ...      |      ...       |     ...")

    # -------------------------------------------------------------------------
    # Test Case (ii): Comparison of 4 stopping criteria + "all" on f1, tol=1e-6
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30 + " TEST CASE (ii) " + "=" * 30)
    criteria = ["absolute", "relative", "residual", "interval", "all"]
    print(f"{'Criterion':<12} | {'Iterations':<10} | {'Root':<14} | {'Reason'}")
    print("-" * 75)
    for crit in criteria:
        res_crit = bisection(f1, 1.0, 2.0, tol=1e-6, error_type=crit)
        print(f"{crit:<12} | {res_crit['iterations']:<10} | {res_crit['root']:<14.8f} | {res_crit['reason']}")

    # -------------------------------------------------------------------------
    # Test Case (iii): Nine Deliberately Triggered Failure Modes & Edge Cases
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30 + " TEST CASE (iii) " + "=" * 30)

    # 1. a >= b
    try:
        bisection(f1, 2.0, 1.0)
    except ValueError as e:
        print(f"[Caught Edge Case 1 (a >= b)]: {e}")

    # 2. f(a) and f(b) have the same sign
    try:
        bisection(lambda x: x**2 - 1, -0.5, 0.5)
    except ValueError as e:
        print(f"[Caught Edge Case 2 (Same Sign)]: {e}")

    # 3. f(a) = 0 exactly (boundary root short-circuit)
    res_fa0 = bisection(lambda x: (x - 1.0) * (x - 3.0), 1.0, 2.5)
    print(f"[Handled Edge Case 3 (f(a)==0)]: Root = {res_fa0['root']}, Iters = {res_fa0['iterations']}, Reason = '{res_fa0['reason']}'")

    # 4. f(b) = 0 exactly (boundary root short-circuit)
    res_fb0 = bisection(lambda x: (x - 1.0) * (x - 3.0), 0.0, 3.0)
    print(f"[Handled Edge Case 4 (f(b)==0)]: Root = {res_fb0['root']}, Iters = {res_fb0['iterations']}, Reason = '{res_fb0['reason']}'")

    # 5. tol <= 0
    try:
        bisection(f1, 1.0, 2.0, tol=-1e-5)
    except ValueError as e:
        print(f"[Caught Edge Case 5 (tol <= 0)]: {e}")

    # 6. max_iter <= 0
    try:
        bisection(f1, 1.0, 2.0, max_iter=0)
    except ValueError as e:
        print(f"[Caught Edge Case 6 (max_iter <= 0)]: {e}")

    # 7. Invalid error_type string
    try:
        bisection(f1, 1.0, 2.0, error_type="foo")
    except ValueError as e:
        print(f"[Caught Edge Case 7 (Invalid error_type)]: {e}")

    # 8. f is not callable
    try:
        bisection("not_callable", 1.0, 2.0)
    except TypeError as e:
        print(f"[Caught Edge Case 8 (f not callable)]: {e}")

    # 9. max_iter reached without convergence (e.g. max_iter=3)
    res_max = bisection(f1, 1.0, 2.0, tol=1e-8, max_iter=3)
    print(f"[Handled Edge Case 9 (max_iter too low)]: Converged = {res_max['converged']}, Reason = '{res_max['reason']}'")

    # -------------------------------------------------------------------------
    # Test Case (iv): f(x) = sin(4*pi*x) on [0, 1]
    # Note: f(0) = sin(0) = 0 and f(1) = sin(4*pi) = 0, so [0, 1] does NOT satisfy f(a)*f(b) < 0.
    # We demonstrate this bracketing failure first, then solve on a valid sub-interval.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30 + " TEST CASE (iv) " + "=" * 30)
    f_sin = lambda x: np.sin(4 * np.pi * x)
    try:
        print("Attempting bisection on full interval [0, 1]:")
        bisection(f_sin, 0.0, 1.0)
    except ValueError as e:
        print(f"Verification: [0, 1] fails opposite sign requirement as expected: {e}")

    # Choose valid sub-bracket containing root at x = 0.25 (e.g. [0.1, 0.3])
    # f(0.1) = sin(0.4*pi) > 0, f(0.3) = sin(1.2*pi) < 0
    sub_a, sub_b = 0.1, 0.3
    res_sin = bisection(f_sin, sub_a, sub_b, tol=1e-6, error_type="interval")
    print(f"\nEvaluating valid sub-bracket [{sub_a}, {sub_b}]:")
    print(f"Root found: {res_sin['root']:.6f} (Theoretical root: 0.25)")
    print(f"Iterations: {res_sin['iterations']}")

    # -------------------------------------------------------------------------
    # Test Case (v): Convergence Plot Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30 + " TEST CASE (v) " + "=" * 30)
    plot_convergence(res1)
    print("All test cases completed successfully.")

        



    