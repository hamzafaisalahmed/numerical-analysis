from math import cos, sin

from multiplicity import auto_newton, estimate_multiplicity, modified_newton

from fixed_point import fixed_point_iteration
from newton import newton

M = 1.0
tol = 1e-10

# van der Waals constants for CO2
A_VDW = 3.592  # atm L^2 / mol^2
B_VDW = 0.04267  # L / mol
R_GAS = 0.082057  # L atm / (mol K)


def fmt(o):
    return f"{o:.3f}" if o is not None else "None"


##------ APPLICATION A ------##


def solveKepler(e, M=M, tol=tol):
    g = lambda E: M + e * sin(E)
    f = lambda E: E - e * sin(E) - M
    fp = lambda E: 1 - e * cos(E)

    fixedPointWay = fixed_point_iteration(g, M, tol=tol)
    newtonWay = newton(f, M, fp, tol=tol)

    return fixedPointWay, newtonWay


def solvingAppQ():
    caelestials = [("Mars", 0.0934), ("Mercury", 0.2056), ("Halley's Comet", 0.967)]

    print(
        f"{'Body':<16}{'e':>7} | {'FP iters':>8}{'FP order':>10} | "
        f"{'Newton iters':>12}{'N order':>9} | {'E':>14}"
    )

    for name, e in caelestials:
        fixed, new = solveKepler(e)
        fp_order = fmt(fixed["order_estimate"]["order"])
        n_order = fmt(new["order_estimate"]["order"])
        root = new["root"]
        root_s = f"{root:.10f}" if root is not None else "None"

        print(
            f"{name:<16}{e:>7.4f} | {fixed['iterations']:>8}"
            f"{fp_order:>10} | "
            f"{new['iterations']:>12}{n_order:>9} | "
            f"{root_s:>14}"
        )


##------ APPLICATION B ------##


def vdw_critical_constants(a=A_VDW, b=B_VDW, R=R_GAS):
    """(a) Critical constants from the van der Waals parameters."""
    Vc = 3 * b
    Pc = a / (27 * b**2)
    Tc = 8 * a / (27 * R * b)
    return Tc, Pc, Vc


def vdw_cubic(Pc, Tc, a=A_VDW, b=B_VDW, R=R_GAS):
    """Return f, f', f'' for f(V) = Pc V^3 - (Pc b + R Tc) V^2 + a V - a b."""
    c2 = Pc * b + R * Tc
    f = lambda V: Pc * V**3 - c2 * V**2 + a * V - a * b
    fp = lambda V: 3 * Pc * V**2 - 2 * c2 * V + a
    fpp = lambda V: 6 * Pc * V - 2 * c2
    return f, fp, fpp


def verify_triple_root(Tc, Pc, Vc, f, fp, fpp, a=A_VDW, b=B_VDW, R=R_GAS):
    """(b) Confirm f(Vc) = f'(Vc) = f''(Vc) = 0 up to floating-point error."""
    c2 = Pc * b + R * Tc
    scales = {
        "f(Vc)": max(abs(Pc * Vc**3), abs(c2 * Vc**2), abs(a * Vc), abs(a * b)),
        "f'(Vc)": max(abs(3 * Pc * Vc**2), abs(2 * c2 * Vc), abs(a)),
        "f''(Vc)": max(abs(6 * Pc * Vc), abs(2 * c2)),
    }
    values = {"f(Vc)": f(Vc), "f'(Vc)": fp(Vc), "f''(Vc)": fpp(Vc)}

    print(f"{'quantity':<10}{'value':>14}{'term scale':>14}{'relative':>12}  ok?")
    for name, val in values.items():
        rel = abs(val) / scales[name]
        print(f"{name:<10}{val:>14.3e}{scales[name]:>14.3e}{rel:>12.2e}  {rel < 1e-12}")
    print(f"f'''(Vc) = 6*Pc = {6 * Pc:.6f}  (nonzero -> multiplicity exactly 3)")


def _report(label, res, Vc):
    order = res["order_estimate"]["order"]
    order_s = f"{order:.3f}" if order is not None else "None"
    root = res["root"]
    err_s = f"{abs(root - Vc):.3e}" if root is not None else "n/a"
    print(
        f"{label:<28} converged={res['converged']!s:<5} iters={res['iterations']:<4} "
        f"order={order_s:<7} |V-Vc|={err_s}"
    )
    if not res["converged"]:
        print(f"    reason: {res['reason']}")


def vdw_application():
    # (a)
    Tc, Pc, Vc = vdw_critical_constants()
    print(f"(a) Tc = {Tc:.6f} K,  Pc = {Pc:.6f} atm,  Vc = {Vc:.6f} L/mol\n")

    f, fp, fpp = vdw_cubic(Pc, Tc)

    # (b)
    print("(b) Triple-root check")
    verify_triple_root(Tc, Pc, Vc, f, fp, fpp)

    V0 = 1.2 * Vc

    # (c) ordinary Newton: expect order -> 1 (error shrinks by ~2/3 per step)
    print(f"\n(c) Ordinary Newton from V0 = 1.2*Vc = {V0:.6f}")
    # Short run for a clean order estimate: stop before rounding noise
    # (~1e-6 in V) takes over. converged=False here is expected.
    res = newton(f, V0, fp, tol=1e-10, max_iter=25)
    _report("newton (25 iters)", res, Vc)

    # Full run, to document what happens near the triple root
    try:
        full = newton(f, V0, fp, tol=1e-10)
        print(
            f"full run: converged={full['converged']}, iters={full['iterations']}, "
            f"reason={full['reason']}"
        )
    except ZeroDivisionError as err:
        print(f"full run raised ZeroDivisionError: {err}")

    # (d) multiplicity estimate + modified Newton
    print("\n(d) Multiplicity detection and modified Newton")
    mult = estimate_multiplicity(f, fp, V0, max_iter=30)
    if mult["m_estimate"] is None:
        print(f"estimate_multiplicity failed: {mult['diagnostic_note']}")
        return
    print(
        f"estimate_multiplicity: m_estimate={mult['m_estimate']:.4f}, "
        f"m_rounded={mult['m_rounded']}, ratio_used={mult['ratio_used']:.4f}"
    )

    res_known = modified_newton(f, V0, fprime=fp, m=mult["m_rounded"], tol=1e-10)
    _report("modified_newton (known m)", res_known, Vc)

    res_u = modified_newton(f, V0, fprime=fp, fprime2=fpp, tol=1e-10)
    _report("modified_newton (u=f/f')", res_u, Vc)

    res_auto = auto_newton(f, fp, fpp, V0)
    _report("auto_newton", res_auto, Vc)
    print(
        f"    method_used={res_auto['method_used']}, "
        f"multiplicity_info={res_auto['multiplicity_info']}"
    )


if __name__ == "__main__":
    solvingAppQ()
    print("\n=== Application B ===")
    vdw_application()
