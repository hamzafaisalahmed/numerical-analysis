from math import cos, sin

from fixed_point import fixed_point_iteration
from newton import newton

M = 1.0
tol = 1e-10


def solveKepler(e, M=M, tol=tol):
    g = lambda E: M + e * sin(E)
    f = lambda E: E - e * sin(E) - M
    fp = lambda E: 1 - e * cos(E)

    fixedPointWay = fixed_point_iteration(g, M, tol=tol)
    newtonWay = newton(f, M, fp, tol=tol)

    return fixedPointWay, newtonWay


def solvingAppQ():
    caelestials = [("Mars", 0.0934), ("Mercury", 0.2056), ("Halley's Commet", 0.967)]

    # printing garbage
    print(
        f"{'Body':<16}{'e':>7} | {'FP iters':>8}{'FP order':>10} | "
        f"{'Newton iters':>12}{'N order':>9} | {'E':>14}"
    )

    for name, e in caelestials:
        fixed, new = solveKepler(e)

        # printing more garbage
        print(
            f"{name:<16}{e:>7.4f} | {fixed['iterations']:>8}"
            f"{fixed['order_estimate']['order']:>10.3f} | "
            f"{new['iterations']:>12}{new['order_estimate']['order']:>9.3f} | "
            f"{new['root']:>14.10f}"
        )


if __name__ == "__main__":
    solvingAppQ()
