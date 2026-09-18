import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

def find_var(expr): #helper function
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

def plot_taylor_approximations(expr, point, orders,x_range, var=None):
    x_arr = np.linspace(x_range[0],x_range[1],300)
    if var == None:
        var = find_var(expr)
    f = sp.lambdify(var,expr)
    y_f = f(x_arr)
    plt.plot(x_arr,y_f,label = "f(x)")
    for order in orders:
        series = taylor_series(expr,point, order,var)
        der = sp.lambdify(var,series["polynomial"])
        y_arr = der(x_arr)
        label = f"P_{order}(x)"
        plt.plot(x_arr,y_arr,label=label)
    plt.legend()
    plt.title(f"Taylor Approximations of {sp.latex(expr)} at x={point}")
    plt.savefig('taylor_plot.png', dpi=300)
    


if __name__ == "__main__":
    x = sp.Symbol('x')
    expr = sp.exp(x)
    print(taylor_series(expr,0,3))
    print(evaluate_and_compare(expr,0,3,1))
    plot_taylor_approximations(expr, 0, [1, 3], [-3, 3])  