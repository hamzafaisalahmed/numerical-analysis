import sympy as sp

def taylor_series ( expr , point , n , var = None ):

    if (type(expr) == str):
        expr = sp.sympify(expr)
    if var == None:
        variables = expr.free_symbols
        if len(variables) != 1:
            raise ValueError("0 or >1 free symbols")
        var = list(variables)[0]
    if type(n) != int or n < 0:
        raise ValueError("n must be a positive integer")

    return_expr = 0
    return_data = {"polynomial":return_expr,
                   "terms" : [],
                   "latex":""}

    for i in range(n+1):
        derivative = expr.diff(var,i).subs(var,point)
        term = derivative/sp.factorial(i) * (var-point)**(i)
        return_expr += term
        return_data["terms"].append(term)
    return_data["polynomial"] = return_expr
    return_data["latex"] = sp.latex(return_expr)
    return return_data

x = sp.Symbol('x')
expr = sp.exp(x)
print(taylor_series(expr,0,3))
