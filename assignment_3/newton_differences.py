import math
import warnings
import numpy as np
# import matplotlib.pyplot as plt


def difference_table(y_nodes):
    """ 3 Build the full forward-difference table. 
    Returns 
    ------
    list of 1-D ndarrays: table[0] = y, table[1] = Delta y, 
    table[2] = Delta^2 y, ..., table[n] (length 1). 
    (The same table also contains all backward differences:  
    nabla^k y_i = Delta^k y_{i-k}.) """

    n = len(y_nodes)
    if n == 0:
        return []
    
    table = [np.array(y_nodes, dtype=float)]  #table[0] is simply y_nodes copied into the table 

    for i in range(1,n):
        previousLevel = table[i-1]
        currentlevel = previousLevel[1:] - previousLevel[:-1]
        table.append(currentlevel)


    return table


def print_difference_table(x_nodes, y_nodes, decimals=6):  
    """Pretty-print the table in the usual triangular textbook layout. """


    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have the same length.")

    n = len(x_nodes)
    table = difference_table(y_nodes)

    headers = ["x", "f(x)"] + ["Δy" if k == 1 else f"Δ^{k}y" for k in range(1, n)]

    grid = np.full((n, n + 1), np.nan)
    for i in range(n):
        grid[i, 0] = x_nodes[i]  # Column 0: x values
        for k in range(n - i):
            grid[i, k + 1] = table[k][i]  # Columns 1+: difference levels
            
    # Format column widths for neat alignment
    col_width = max(12, decimals + 6)
    
    # Print headers
    header_row = "".join(f"{h:>{col_width}}" for h in headers)
    print(header_row)
    print("-" * len(header_row))
    
    # Print rows
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
    """ Return h if the sorted nodes are equally spaced (within the relative 
    tolerance rtol), otherwise raise ValueError reporting the largest 
    deviation from uniform spacing."""

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



def newton_forward(x_nodes, y_nodes, x_eval, degree=None):
    """ Newton forward-difference interpolation.
    If degree is None, use all nodes. Otherwise use degree+1 
    consecutive nodes, starting from the base node x_k = the largest  
    node <= x_eval for which degree+1 nodes x_k, ..., x_{k+degree} 
    still exist (so the formula is used where it is most accurate).
    Returns
    ------
    dict with keys: "value", "degree", "base_node", "s", 
    "terms"             -- list of the individual terms added 
    "error_estimate"    -- magnitude of the first OMITTED term 
                            (None if no higher difference exists) 
    Raises
    -----
    ValueError          -- unequally spaced nodes, degree too large, 
                            mismatched lengths, fewer than 2 nodes
    """
    if np.ndim(x_eval) > 0:
        return np.array([newton_forward(x_nodes, y_nodes, xe, degree)["value"]
                         for xe in x_eval])

    h = check_equal_spacing(x_nodes)  # also checks fewer than 2 nodes and unequally spaces nodes
   

    n = len(x_nodes)

    if n != len(y_nodes):
        raise ValueError("Lengths mismataches for x and y ")
    
    table =  difference_table(y_nodes) 
    if degree is None:
        k = 0 
        deg = n -1
    else:
        if(degree < 1 or degree > n-1):
            raise ValueError("Given Degree is either too large or too small")
        else:
            deg = degree
            k = 0
            for i in range(n):
                if(x_eval >= x_nodes[i]):
                    k = i
                else:
                    break


    if (k + deg) >= n :
        k = n - 1 -deg

    



    s = (x_eval - x_nodes[k]) / h

        

    terms = []
    term_0 = table[0][k]
    terms.append(term_0)
    value = term_0

    s_prod = 1
    factorial = 1
    for j in range(1, deg+1):
        s_prod *=  s-(j-1)
        factorial *= j
        term_j = (s_prod/factorial) * table[j][k]
        terms.append(term_j)
        value += term_j

    error_estimate = None
    nl = deg+1
    if(k + nl ) < n:
        error_estimate = abs(((s_prod * (s-deg))/(factorial * nl)) * (table[nl][k]))

    return {
        "value": value,
        "degree": deg,
        "base_node": x_nodes[k],
        "s": s,
        "terms": terms,
        "error_estimate": error_estimate
    }


def newton_backward(x_nodes, y_nodes, x_eval, degree=None): 
    """ Newton backward-difference interpolation; same return format as 
    newton_forward, with "s" replaced by "t" and the base node chosen 
    as the smallest node >= x_eval for which degree+1 nodes 
    x_{k-degree}, ..., x_k exist. """

    if np.ndim(x_eval) > 0:
        return np.array([newton_backward(x_nodes, y_nodes, xe, degree)["value"]
                         for xe in x_eval])

    h = check_equal_spacing(x_nodes)  # also checks fewer than 2 nodes and unequally spaces nodes
   

    n = len(x_nodes)

    if n != len(y_nodes):
        raise ValueError("Lengths mismataches for x and y ")
    
    table =  difference_table(y_nodes) 
    if degree is None:
        k = n-1 
        deg = n -1
    else:
        if(degree < 1 or degree > n-1):
            raise ValueError("Given Degree is either too large or too small")
        else:
            deg = degree
            k = n-1
            for i in range(n-1 , -1,-1):
                if(x_eval <= x_nodes[i]):
                    k = i
                else:
                    break


    if (k - deg) < 0 :
        k = deg


    t = (x_eval - x_nodes[k]) / h
   
    terms = []
    term_0 = table[0][k]
    terms.append(term_0)
    value = term_0

    t_prod = 1
    factorial = 1
    for j in range(1, deg+1):
        t_prod *=  t+(j-1)
        factorial *= j
        term_j = (t_prod/factorial) * table[j][k-j]
        terms.append(term_j)
        value += term_j

    error_estimate = None
    nl = deg+1
    if(k - nl ) >= 0 :
        error_estimate = abs(((t_prod * (t+deg))/(factorial * nl)) * (table[nl][k-nl]))

    return {
        "value": value,
        "degree": deg,
        "base_node": x_nodes[k],
        "t": t,
        "terms": terms,
        "error_estimate": error_estimate
    }

    
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

    print("\n=== Error handling ===")
    for label, call in [
        ("unequal spacing", lambda: newton_forward([0, 1, 2.5, 3], [0, 1, 2, 3], 1)),
        ("degree too big ", lambda: newton_forward(x, y, 0.3, degree=6)),
        ("length mismatch", lambda: newton_backward([0, 1, 2], [0, 1], 1)),
        ("one node       ", lambda: newton_forward([0], [1], 0)),
    ]:
        try:
            call()
        except ValueError as e:
            print(f"{label}: ValueError -> {e}")

    print("\n=== Extrapolation warning ===")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        newton_forward(x, y, 1.2, degree=2)
        print("warning raised:", bool(w))