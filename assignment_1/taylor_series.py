import sympy as sp

def find_var(expr):
    variables = expr.free_symbols
    if len(variables) != 1:
        raise ValueError("0 or >1 free symbols")
    return list(variables)[0]

def taylor_series ( expr , point , n , var = None ):

    if (type(expr) == str):
        expr = sp.sympify(expr)
    if var == None:
        var = find_var(expr)
    if type(n) != int or n < 0:
        raise ValueError("n must be a positive integer")

    return_expr = 0
    return_data = {"polynomial":return_expr,
                   "terms" : [],
                   "remainder" : return_expr,
                   "latex":""}

    for i in range(n+1):
        derivative = expr.diff(var,i).subs(var,point)
        term = derivative/sp.factorial(i) * (var-point)**(i)
        return_expr += term
        return_data["terms"].append(term)
    return_data["polynomial"] = return_expr
    return_data["latex"] = sp.latex(return_expr)

    xi = sp.symbols('xi')
    return_data["remainder"] = (expr.diff(var,n+1).subs(var,xi) / sp.factorial(n+1) ) * (var-point)**(n+1)

    return return_data

def evaluate_and_compare(expr, point, n, x_eval, var=None):
    if var == None:
            var=find_var(expr)
    series = taylor_series(expr,point,n,var)

    approx = sp.N(series["polynomial"].subs(var,x_eval))
    true_value = sp.N(expr.subs(var,x_eval))
    abs_error = abs(true_value - approx)
    rel_error = abs_error/true_value
    return {"approx" : approx, "true_value" : true_value, "abs_error" : abs_error, "rel_error" : rel_error}



if __name__ == "__main__":
    x = sp.Symbol('x')
    expr = sp.exp(x)
    print(taylor_series(expr,0,3))
    print(evaluate_and_compare(expr,0,3,1))
