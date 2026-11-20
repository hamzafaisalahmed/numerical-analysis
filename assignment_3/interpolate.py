import lagrange, hermite, newton_differences
import numpy as np

def sort_inputs(x_nodes,y_nodes,dydx_nodes=None):
    """
    Sort the nodes by increasing x, keeping y (and dy/dx) paired with them.

    Parameters
    ----------
    x_nodes : list/array
        Nodes, in any order.
    y_nodes : list/array
        Function values at the nodes.
    dydx_nodes : list/array, optional
        Derivative values at the nodes.

    Returns
    -------
    x_sorted, y_sorted, dydx_sorted : arrays
        Sorted data (dydx_sorted is None if no derivatives were given).

    Raises
    ------
    ValueError
        If the input lists have different lengths.

    Example
    -------
    >>> x, y, d = sort_inputs([2, 0, 1], [4, 0, 1])
    >>> x
    array([0., 1., 2.])
    >>> y
    array([0., 1., 4.])
    >>> d is None
    True
    """
    if len(x_nodes) != len(y_nodes):
            raise ValueError("x_nodes and y_nodes must have the same length.")
    if dydx_nodes is not None and len(dydx_nodes) != len(x_nodes):
        raise ValueError("dydx_nodes must have the same length as x_nodes.")
    sortedArr = np.argsort(x_nodes)
    x_sorted = []
    y_sorted = []
    if dydx_nodes is None:
        dydx_sorted = None
    else:
        dydx_sorted = []
    for idx in sortedArr:
        x_sorted.append(x_nodes[idx])
        y_sorted.append(y_nodes[idx])
        if (dydx_nodes is not None):
            dydx_sorted.append(dydx_nodes[idx])

    x_sorted = np.array(x_sorted,dtype=float)
    y_sorted = np.array(y_sorted, dtype=float)
    if dydx_sorted is not None:
        dydx_sorted = np.array(dydx_sorted, dtype=float)
    
    return x_sorted,y_sorted,dydx_sorted

def choose_method(x_nodes, y_nodes, x_eval, dydx_nodes=None,
                  spacing_rtol=1e-9):
    """
    Inspect the data and decide which interpolation method to use.

    Decision rules (in this order):
      1. If dydx_nodes is supplied (not None) -> "hermite"
      2. Else, if the nodes are equally spaced:
           compute s = (x_eval - x_0)/h for the sorted nodes.
           If s <= n/2 (x_eval in the first half of the table,
           or to the left of it) -> "newton_forward"
           otherwise (second half, or right of it) -> "newton_backward"
      3. Else (unequally spaced) -> "lagrange"

    Parameters
    ----------
    x_nodes : list/array
        Nodes, in any order.
    y_nodes : list/array
        Function values (not used in the decision).
    x_eval : float
        Point to evaluate at.
    dydx_nodes : list/array, optional
        Derivative values. If given, Hermite is chosen.
    spacing_rtol : float, default 1e-9
        Tolerance for deciding the node gaps are equal.

    Returns
    -------
    dict: {"method": str,
           "reason": str,
           "equally_spaced": bool, "h": float or None,
           "extrapolating": bool}

    Raises
    ------
    TypeError
            if x_eval is not an int or float.
    
    Example
    -------
    >>> x = [0, 1, 2, 3, 4, 5]
    >>> choose_method(x, x, 0.4)["method"]
    'newton_forward'
    """
    if isinstance(x_eval, np.ndarray):
        if x_eval.size != 1 or x_eval.ndim != 1:
            raise TypeError("x_eval must be a single number.")
        else:
            x_eval = float(x_eval[0])
    elif not isinstance(x_eval, (int, float,np.integer, np.floating)):
            raise TypeError("x_eval must be a single number.")
    x_sorted,_,_ = sort_inputs(x_nodes, y_nodes, dydx_nodes)
    method = ""
    reason = ""
    equally_spaced = False
    h = None
    extrapolating = False
    n = len(x_nodes)-1

    try:
        h = newton_differences.check_equal_spacing(x_sorted,spacing_rtol)
        equally_spaced = True
    except ValueError:
        h = None
    if (x_eval < x_sorted[0]) or (x_eval > x_sorted[-1]):
        extrapolating = True
    if dydx_nodes is not None:
        method = "hermite"
        reason = "derivatives given, so hermite is used"
    elif h is None:
        method = "lagrange"
        reason = "nodes not equally spaced and no derivative provided so used lagrange"
    else:
        s = (x_eval-x_sorted[0])/h
        if s<=n/2:
            method = "newton_forward"
            reason = f"nodes equally spaced with h={h:g}; x_eval lies near the left of the table (s={s:.3g} of n={n})"
        else:
            method = "newton_backward"
            reason = f"nodes equally spaced with h={h:g}; x_eval lies near the right of the table (s={s:.3g} of n={n})"
        
    return {"method": method, "reason": reason, "equally_spaced": equally_spaced, "h" : h, "extrapolating": extrapolating}

def interpolate(x_nodes, y_nodes, x_eval, dydx_nodes=None,
                degree=None, verbose=False):
    """
    Sort the data, pick a method with choose_method(), and evaluate it.

    Parameters
    ----------
    x_nodes : list/array
        Nodes, in any order.
    y_nodes : list/array
        Function values at the nodes.
    x_eval : float/int
        Point to evaluate at.
    dydx_nodes : list/array, optional
        Derivative values. If given, Hermite is used.
    degree : int, optional
        Degree for the Newton formulas (ignored by Lagrange and Hermite).
    verbose : bool, default False
        Print the method, value and reason.

    Returns
    -------
    dict
        The chosen method's result dict, plus "method_used" and
        "decision" (the dict from choose_method).

    Raises
    ------
    ValueError
        if the method returned by choice is not lagrange, hermite, newton_forward or newton_back.
    
    TypeError
        if x_eval is not an int or float.

    Example
    -------
    >>> x = [5, 4, 3, 2, 1, 0]
    >>> interpolate(x, x, 0.4)["method_used"]
    'newton_forward'
    """
    if isinstance(x_eval, np.ndarray):
        if x_eval.size != 1 or x_eval.ndim != 1:
            raise TypeError("x_eval must be a single number.")
        else:
            x_eval = float(x_eval[0])
    elif not isinstance(x_eval, (int, float,np.integer, np.floating)):
                raise TypeError("x_eval must be a single number.")
    x_sorted, y_sorted, dydx_sorted = sort_inputs(x_nodes, y_nodes, dydx_nodes)
    choice = choose_method(x_sorted, y_sorted, x_eval, dydx_sorted)
    if choice["method"] == "lagrange":
        result = lagrange.lagrange_interpolate(x_sorted, y_sorted, x_eval)
    elif choice["method"] == "newton_forward":
        result = newton_differences.newton_forward(x_sorted, y_sorted, x_eval, degree)
    elif choice["method"] == "newton_backward":
        result = newton_differences.newton_backward(x_sorted, y_sorted, x_eval, degree)
    elif choice["method"] == "hermite":
        result = hermite.hermite_interpolate(x_sorted, y_sorted, dydx_sorted,x_eval)
    else:
        raise ValueError(f"Unknown method: {choice['method']}")
    
    result["method_used"] = choice["method"]
    result["decision"] = choice
    
    if verbose:
        d = choice
        h = "n/a" if d["h"] is None else f"{d['h']:g}"
        print("-" * 70)
        print(f"  method         : {d['method']}")
        print(f"  value          : {float(result['value']):.8f}")
        print(f"  degree         : {result.get('degree', 'n/a')}")
        print(f"  equally spaced : {d['equally_spaced']}  (h = {h})")
        print(f"  extrapolating  : {d['extrapolating']}")
        print(f"  reason         : {d['reason']}\n")
    return result


if __name__ == "__main__":
    
    x_nodes = [
    [0, 1, 2, 3, 4, 5],            
    [0, 1, 2, 3, 4, 5],            
    [0, 0.3, 1.1, 2.0, 3.5],       
    [0, 0.3, 1.1, 2.0, 3.5],       
    [5, 4, 3, 2, 1, 0],            
    [0, 0.1, 0.2000000001, 0.3],   
    [0, 1, 2, 3, 4, 5],            
    ]

    x_eval = [0.4, 4.7, 1.5, 1.5, 0.4, 0.15, 6.5]
    for i in range(len(x_nodes)):
        y = np.sin(x_nodes[i])
        if i == 3: #T4 the only one with derivative provided
            dydx_nodes = np.cos(x_nodes[i])
        else:
            dydx_nodes = None
        print(f"T{i+1}   x_eval = {x_eval[i]}")
        r = interpolate(x_nodes[i], y, x_eval[i],dydx_nodes,verbose = True)
