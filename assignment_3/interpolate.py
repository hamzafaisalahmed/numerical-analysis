import lagrange, hermite, newton_differences
import numpy as np

def sort_inputs(x_nodes,y_nodes,dydx_nodes=None):

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

    Returns
    -------
    dict: {"method": str,
           "reason": str (a human-readable sentence, e.g.
               "nodes equally spaced with h=10; x_eval=2015 lies
                near the end of the table (s=6.5 of n=7)"),
           "equally_spaced": bool, "h": float or None,
           "extrapolating": bool}
    """
    method = ""
    reason = ""
    equally_spaced = False
    h = None
    extrapolating = False
    n = len(x_nodes)

    try:
        h = newton_differences.check_equal_spacing(x_nodes,spacing_rtol)
        equally_spaced = True
    except ValueError:
        h = None
    if (x_eval < x_nodes[0]) or (x_eval > x_nodes[n-1]):
        extrapolating = True
    if dydx_nodes is not None:
        method = "hermite"
        reason = "dydx_nodes are available so hermite is usable"
    elif h is None:
        method = "lagrange"
        reason = "dydx not available, data not equally spaced"
    else:
        s = (x_eval-x_nodes[0])/h
        if s<=n/2:
            method = "newton_forward"
            reason = f"nodes equally spaced with h ={h}, x_eval lies closer to the left of table"
        else:
            method = "newton_backward"
            reason = f"nodes equally spaced with h ={h}, x_eval lies closer to the right of table"
        
    return {"method": method, "reason": reason, "equally_spaced": equally_spaced, "h" : h, "extrapolating": extrapolating}

def interpolate(x_nodes, y_nodes, x_eval, dydx_nodes=None,
                degree=None, verbose=False):
    """
    Call choose_method, then dispatch to lagrange_interpolate,
    newton_forward, newton_backward or hermite_interpolate.
    `degree` is passed to the Newton formulas (ignored otherwise).

    Returns
    -------
    the chosen method's result dict, PLUS the keys
    "method_used" and "decision" (the dict from choose_method)
    """
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

    result["method_used"] = choice["method"]
    result["decision"] = choice
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
        if i == 3:
            dydx_nodes = np.cos(x_nodes[i])
        else:
            dydx_nodes = None
        r = interpolate(x_nodes[i], y, x_eval[i],dydx_nodes)
        print(f"T{i+1}: {r['method_used']:<15} value={r['value']:.8f} | h={r['decision']['h']} | equally_spaced={r['decision']['equally_spaced']} | extrapolating={r['decision']['extrapolating']} | {r['decision']['reason']}\n")