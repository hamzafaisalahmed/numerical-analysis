import numpy as np
import scipy.optimize as opt
import sympy as sp
import matplotlib.pyplot as plt

print("--- Testing installed libraries ---")

# 1. SymPy Test (Symbolic Math)
x = sp.Symbol('x')
symbolic_expr = sp.sin(x) * sp.exp(x)
derivative = sp.diff(symbolic_expr, x)
print(f"[SymPy] Derivative of sin(x)*exp(x) with respect to x: {derivative}")

# 2. NumPy Test (Array Operations)
x_vals = np.linspace(-3, 3, 200)
y_vals = np.exp(-x_vals**2)  # Gaussian curve
print(f"[NumPy] Generated array of shape {x_vals.shape} with values ranging from {x_vals.min()} to {x_vals.max()}")

# 3. SciPy Test (Numerical Optimization / Peak Finding)
# Find the maximum of -y_vals (i.e., peak of the Gaussian curve at x = 0)
def target_func(x):
    return -np.exp(-x**2)

res = opt.minimize_scalar(target_func)
print(f"[SciPy] Peak detected near x = {res.x:.4f} with value = {-res.fun:.4f}")

# 4. Matplotlib Test (Plotting)
print("[Matplotlib] Displaying graph window...")
plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals, label=r"$f(x) = e^{-x^2}$", color="crimson", linewidth=2)
plt.axvline(x=res.x, color="gray", linestyle="--", label=f"SciPy Peak (x={res.x:.2f})")
plt.title("Scientific Stack Test: NumPy, SciPy, SymPy & Matplotlib")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Display the window
plt.show()

print("--- All 4 libraries executed successfully! ---")