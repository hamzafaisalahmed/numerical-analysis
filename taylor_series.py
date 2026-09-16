import sympy as sp

def taylor_series ( expr , point , n , var = None ):

    if n < 0:
        return ValueError
    return_expr = 0
    for i in range(n+1):
        derivative = expr.diff(var,i).subs(var,point)
        term = derivative/sp.factorial(i) * (var-point)**(i)
        return_expr += term
    return return_expr

x = sp.Symbol('x')
expr = sp.exp(x)
print(taylor_series(expr,0,3,x))
