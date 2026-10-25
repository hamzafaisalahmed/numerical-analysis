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
    """
    Format an order-of-convergence value for table output.

    Parameters
    ----------
    o : float or None
        Value to format, typically an "order" from estimate_order().

    Returns
    -------
    str
        The value to 3 decimal places, or "None" if o is None.

    Raises
    ------
    ValueError or TypeError
        If o is neither None nor a number.

    Examples
    --------
    >>> fmt(1.99412)
    '1.994'
    >>> fmt(None)
    'None'
    """
    return f"{o:.3f}" if o is not None else "None"


##------ APPLICATION A ------##


def solveKepler(e, M=M, tol=tol):
    """
    Solve Kepler's equation M = E - e*sin(E) for E two ways, both from E0 = M.

    Parameters
    ----------
    e : float
        Orbital eccentricity, 0 <= e < 1.
    M : float, optional
        Mean anomaly in radians (default 1.0).
    tol : float, optional
        Stopping tolerance for both solvers (default 1e-10).

    Returns
    -------
    tuple of dict
        (fixed_point_result, newton_result), each in the standard solver
        format: "root", "iterations", "converged", "reason", "history",
        "order_estimate".

    Raises
    ------
    ValueError
        Propagated from the solvers if tol <= 0.
    ZeroDivisionError
        Propagated from newton() if |1 - e*cos(E)| < 1e-12 (cannot occur
        for e < 1).

    Examples
    --------
    >>> fixed, new = solveKepler(0.0934)
    >>> round(new["root"], 6)
    1.082484
    """
    g = lambda E: M + e * sin(E)
    f = lambda E: E - e * sin(E) - M
    fp = lambda E: 1 - e * cos(E)

    fixedPointWay = fixed_point_iteration(g, M, tol=tol)
    newtonWay = newton(f, M, fp, tol=tol)

    return fixedPointWay, newtonWay


def solvingAppQ():
    """
    Application A driver: solve Kepler's equation for three bodies and print a table.

    For Mars (e = 0.0934), Mercury (e = 0.2056) and Halley's Comet (e = 0.967),
    with M = 1.0 rad and tol = 1e-10, solves for E by fixed-point iteration and
    by Newton's Method via solveKepler(), then prints iteration counts, empirical
    orders of convergence and the converged E.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Results are printed to standard output.

    Raises
    ------
    ZeroDivisionError
        Propagated from newton() if |f'(E)| falls below its threshold
        (cannot occur for e < 1).

    Example
    --------
    solvingAppQ() 
    """
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
    """
    Compute the van der Waals critical constants from the a and b parameters.

    Uses Vc = 3b, Pc = a / (27 b^2), Tc = 8a / (27 R b).

    Parameters
    ----------
    a : float, optional
        Attraction parameter in atm L^2 / mol^2 (default: CO2, 3.592).
    b : float, optional
        Excluded volume in L / mol (default: CO2, 0.04267).
    R : float, optional
        Gas constant in L atm / (mol K) (default 0.082057).

    Returns
    -------
    tuple of float
        (Tc, Pc, Vc) in K, atm and L/mol.

    Raises
    ------
    ZeroDivisionError
        If b == 0 or R == 0.
    TypeError
        If a, b or R is not a number.

    Examples
    --------
    >>> Tc, Pc, Vc = vdw_critical_constants()
    >>> round(Tc, 3), round(Pc, 3), round(Vc, 5)
    (303.965, 73.068, 0.12801)
    """
    Vc = 3 * b
    Pc = a / (27 * b**2)
    Tc = 8 * a / (27 * R * b)
    return Tc, Pc, Vc


def vdw_cubic(Pc, Tc, a=A_VDW, b=B_VDW, R=R_GAS):
    """
    Build the van der Waals cubic in molar volume and its first two derivatives.

    f(V)   = Pc V^3 - (Pc b + R Tc) V^2 + a V - a b
    f'(V)  = 3 Pc V^2 - 2 (Pc b + R Tc) V + a
    f''(V) = 6 Pc V - 2 (Pc b + R Tc)

    Parameters
    ----------
    Pc : float
        Critical pressure in atm.
    Tc : float
        Critical temperature in K.
    a : float, optional
        Attraction parameter in atm L^2 / mol^2 (default: CO2).
    b : float, optional
        Excluded volume in L / mol (default: CO2).
    R : float, optional
        Gas constant in L atm / (mol K) (default 0.082057).

    Returns
    -------
    tuple of callable
        (f, fp, fpp), each taking a molar volume V and returning a float.

    Raises
    ------
    TypeError
        If any parameter is not a number.

    Examples
    --------
    >>> Tc, Pc, Vc = vdw_critical_constants()
    >>> f, fp, fpp = vdw_cubic(Pc, Tc)
    >>> abs(f(Vc)) < 1e-12
    True
    """
    c2 = Pc * b + R * Tc
    f = lambda V: Pc * V**3 - c2 * V**2 + a * V - a * b
    fp = lambda V: 3 * Pc * V**2 - 2 * c2 * V + a
    fpp = lambda V: 6 * Pc * V - 2 * c2
    return f, fp, fpp


def verify_triple_root(Tc, Pc, Vc, f, fp, fpp, a=A_VDW, b=B_VDW, R=R_GAS):
    """
    Check numerically that Vc is a triple root: f(Vc) = f'(Vc) = f''(Vc) = 0.

    Each value is compared against the largest term that cancels in its
    expression, so the test is relative: a quantity passes if
    |value| / term_scale < 1e-12. A table is printed, followed by f'''(Vc) = 6 Pc
    to show the multiplicity is exactly 3.

    Parameters
    ----------
    Tc : float
        Critical temperature in K.
    Pc : float
        Critical pressure in atm.
    Vc : float
        Critical molar volume in L/mol.
    f, fp, fpp : callable
        The cubic and its first and second derivatives, from vdw_cubic().
    a : float, optional
        Attraction parameter (default: CO2).
    b : float, optional
        Excluded volume (default: CO2).
    R : float, optional
        Gas constant (default 0.082057).

    Returns
    -------
    None
        The table is printed to standard output.

    Raises
    ------
    ZeroDivisionError
        If a term scale is zero (not the case for physical parameters).
    TypeError
        If f, fp or fpp is not callable.

    Examples
    --------
    verify_triple_root(Tc, Pc, Vc, f, fp, fpp) 
    """
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
    """
    Print a one-line summary of a root-finder result for the van der Waals runs.

    Parameters
    ----------
    label : str
        Name of the method, printed at the start of the line.
    res : dict
        Result dict from newton(), modified_newton() or auto_newton(), with
        keys "root", "converged", "iterations", "reason" and "order_estimate".
    Vc : float
        Exact critical volume, used to compute |V - Vc|.

    Returns
    -------
    None
        Prints converged flag, iteration count, order estimate and error
        (or "n/a" if root is None). If the run did not converge, the reason
        is printed on a second line.

    Raises
    ------
    KeyError
        If res lacks one of the expected keys.

    Examples
    --------
    _report("newton (20 iters)", res, Vc)
    """
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
    """
    Application B driver: the van der Waals critical point of CO2 (triple root).

    Runs, in order: (a) critical constants, (b) the triple-root check,
    (c) ordinary Newton from V0 = 1.2 Vc (a 20-iteration run for a clean order
    estimate, then a full run showing how it ends), and (d) estimate_multiplicity,
    modified_newton with the known m, modified_newton with u = f/f', and
    auto_newton, each summarised with _report().

    Parameters
    ----------
    None

    Returns
    -------
    None
        Results are printed to standard output. Returns early, after printing
        a message, if estimate_multiplicity cannot produce an estimate.

    Raises
    ------
    ZeroDivisionError
        If a derivative guard fires in the 20-iteration Newton run, the
        modified runs or auto_newton. The full ordinary-Newton run is the
        one place it is caught and reported instead of propagated.

    Examples
    --------
    vdw_application()
    """
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
    res = newton(f, V0, fp, tol=1e-10, max_iter=20)
    _report("newton (20 iters)", res, Vc)

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
    print("\n=== Application B ===\n")
    vdw_application()
