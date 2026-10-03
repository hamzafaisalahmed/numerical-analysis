import math

def estimate_order(sequence, true_value=None):
    """
    Estimate empirical order of convergence alpha from a sequence of iterates.

    Parameters
    ----------
    sequence : list of float
        Sequence of iterates p_0, p_1, p_2, ...
    true_value : float, optional
        Exact fixed point p, if known.

    Returns
    -------
    dict
        {"order": float or None, "note": str}
    """
    if len(sequence) < 3:
        return {"order": None, "note": "Fewer than 3 points provided."}

    
    pts = [float(x) for x in sequence]

    if true_value is not None:
        errors = [abs(p - float(true_value)) for p in pts]
        floor = 1e-7
    else:
        errors = [abs(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        floor = 1e-12 

    clean_errors = [] #filter out errors
    for e in errors:
        if e < floor or math.isnan(e) or math.isinf(e):
            break
        if clean_errors and e >= clean_errors[-1]:
            break
        clean_errors.append(e)

    if len(clean_errors) < 3:
        return {"order": None, "note": "Fewer than 3 usable errors above noise floor."}

    e1 = clean_errors[-3]
    e2 = clean_errors[-2]
    e3 = clean_errors[-1]

    ratio1 = e2 / e1
    ratio2 = e3 / e2
    denom = math.log(ratio1)
    if abs(denom) < 1e-12:
        return {"order": None, "note": "Division by zero."}

    alpha = math.log(ratio2) / denom
    return {"order": alpha, "note": "estimated successfully"}

def fixed_point_iteration(g, p0, tol=1e-8, max_iter=100, verbose=False):
    """
    Locate a fixed point of g starting from p0 using p_{n+1} = g(p_n).

    Parameters
    ----------
    g : callable
        Function g(x).
    p0 : float
        Initial guess.
    tol : float, optional
        Tolerance for stopping criterion (default 1e-8).
    max_iter : int, optional
        Maximum iterations allowed (default 100).
    verbose : bool, optional
        If True, prints progress at each step (default False).

    Returns
    -------
    dict with keys:
        "root" : float or None
        "iterations" : int
        "converged" : bool
        "reason" : str
        "history" : list of dicts: {"n": n, "p": p, "diff": diff}
        "order_estimate" : dict
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if not callable(g):
        raise TypeError("g must be callable")

    p = float(p0)
    history = [{"n": 0, "p": p, "diff": None}]
    converged = False
    reason = f"Reached maximum iterations {max_iter} without convergence"
    root = None

    if verbose:
        print(f"Iteration 0: p = {p}")

    for n in range(1, max_iter + 1):
        try:
            p_next = float(g(p))
        except Exception as e:
            reason = f"Evaluation failed at step {n}: {e}"
            break

        if math.isnan(p_next) or math.isinf(p_next):
            reason = "Diverged"
            break

        diff = abs(p_next - p)
        history.append({"n": n, "p": p_next, "diff": diff})

        if verbose:
            print(f"Iteration {n}: p = {p_next}, diff = {diff}")

        if abs(p_next) > 1e10:
            reason = f"|p_{n}| exceeded 1e10 and diverged"
            break

        if diff < tol: #check convergence
            converged = True
            root = p_next
            reason = f"Converged within tolerance in {n} iterations"
            break

        p = p_next

    p_values = []
    for item in history:
        p_values.append(item["p"])
    order_est = estimate_order(p_values)

    return {
        "root": root,
        "iterations": len(history) - 1,
        "converged": converged,
        "reason": reason,
        "history": history,
        "order_estimate": order_est
    }





if __name__ == "__main__":
    p0 = 1.5
    true_root = 1.365230013

    test_cases = {
        "g1(x) = x - x^3 - 4x^2 + 10": lambda x: x - x**3 - 4*x**2 + 10,
        "g2(x) = sqrt(10/x - 4x)": lambda x: math.sqrt(10/x - 4*x),
        "g3(x) = 0.5 * sqrt(10 - x^3)": lambda x: 0.5 * math.sqrt(10 - x**3),
        "g4(x) = sqrt(10 / (4 + x))": lambda x: math.sqrt(10 / (4 + x)),
        "g5(x) = x - (x^3 + 4x^2 - 10)/(3x^2 + 8x)": lambda x: x - (x**3 + 4*x**2 - 10)/(3*x**2 + 8*x),
    }

    print("Problem 1 Test Cases (p0 = 1.5):")
    for name, g in test_cases.items():
        res = fixed_point_iteration(g, p0)
        
        # Estimate order using true root
        p_list = [h["p"] for h in res["history"]]
        order_res = estimate_order(p_list, true_value=true_root)

        print(f"\n{name}")
        print(f"  Converged : {res['converged']}")
        print(f"  Iterations: {res['iterations']}")
        print(f"  Root      : {res['root']}")
        print(f"  Order     : {order_res['order']}")
        print(f"  Reason    : {res['reason']}")
