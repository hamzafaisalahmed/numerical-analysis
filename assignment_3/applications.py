import numpy as np
import matplotlib.pyplot as plt

from lagrange import check_data, lagrange_interpolate
from newton_differences import (
    difference_table,
    print_difference_table,
    newton_forward,
    newton_backward,
)
from hermite import hermite_interpolate
from interpolate import choose_method, interpolate


def evaluate_census_interpolation(x_nodes, y_nodes, x_eval, degree=3):
    """
    Estimate population at x_eval using automatic method selection and Newton
    forward/backward differences of a given degree.

    Parameters
    ----------
    x_nodes : list or 1-D array
        Census years (equally spaced).
    y_nodes : list or 1-D array
        Population counts corresponding to x_nodes.
    x_eval : float or 1-D numpy array
        Year(s) at which to estimate the population.
    degree : int, default 3
        Polynomial degree for Newton forward/backward interpolation.

    Returns
    -------
    dict
        If x_eval is a scalar, returns the full result dict from interpolate()
        (including "value", "degree", "base_node", "terms", "error_estimate",
        "method_used", and "decision").
        If x_eval is a 1-D array, returns a dict with "value" (1-D ndarray of
        estimates) and "decisions" (list of decision dicts).

    Raises
    ------
    TypeError
        If inputs are non-numeric.
    ValueError
        If x_nodes and y_nodes have mismatched lengths, fewer than 2 nodes,
        are not equally spaced, or degree is out of range.

    Example
    -------
    >>> years = [1950, 1960, 1970, 1980]
    >>> pop = [151.3, 179.3, 203.3, 226.5]
    >>> round(float(evaluate_census_interpolation(years, pop, 1955, degree=3)["value"]), 4)
    166.0
    """
    x_arr, y_arr = check_data(x_nodes, y_nodes)
    if np.ndim(x_eval) > 0:
        x_eval_arr = np.asarray(x_eval, dtype=float)
        values = []
        decisions = []
        for xe in x_eval_arr:
            res = interpolate(x_arr, y_arr, float(xe), degree=degree)
            values.append(float(res["value"]))
            decisions.append(res["decision"])
        return {"value": np.array(values, dtype=float), "decisions": decisions}

    return interpolate(x_arr, y_arr, float(x_eval), degree=degree)


def evaluate_steam_table(x_nodes, y_nodes, x_eval, degrees=(1, 2, 3, 4), ref_value=None):
    """
    Estimate saturation pressure at x_eval across multiple polynomial degrees
    using automatic forward/backward selection.

    Parameters
    ----------
    x_nodes : list or 1-D array
        Tabulated temperatures T (deg C).
    y_nodes : list or 1-D array
        Tabulated saturation pressures P_sat (kPa).
    x_eval : float or 1-D numpy array
        Temperature(s) at which to evaluate P_sat.
    degrees : tuple/list of int, default (1, 2, 3, 4)
        Degrees of Newton interpolation to compare when x_eval is a scalar.
    ref_value : float, optional
        Reference saturation pressure at scalar x_eval for relative error calculation.

    Returns
    -------
    dict
        If x_eval is a scalar:
            {"decision": decision dict from choose_method,
             "results": list of dicts for each degree with keys "degree",
                        "method_used", "value", "error_estimate", "abs_error",
                        "rel_error", "base_node"}
        If x_eval is a 1-D array:
            {"value": 1-D ndarray evaluated with highest degree in degrees}

    Raises
    ------
    TypeError
        If inputs are non-numeric.
    ValueError
        If nodes are not equally spaced, lengths mismatch, or degree is invalid.

    Example
    -------
    >>> T = [20, 30, 40, 50, 60]
    >>> P = [2.3392, 4.2469, 7.3851, 12.352, 19.947]
    >>> res = evaluate_steam_table(T, P, 25.0, degrees=(1, 2))
    >>> len(res["results"])
    2
    """
    x_arr, y_arr = check_data(x_nodes, y_nodes)
    if np.ndim(x_eval) > 0:
        x_eval_arr = np.asarray(x_eval, dtype=float)
        deg = max(degrees) if degrees else None
        vals = np.array(
            [float(interpolate(x_arr, y_arr, float(xe), degree=deg)["value"]) for xe in x_eval_arr],
            dtype=float,
        )
        return {"value": vals}

    xe = float(x_eval)
    decision = choose_method(x_arr, y_arr, xe)
    results = []
    for d in degrees:
        r = interpolate(x_arr, y_arr, xe, degree=d)
        val = float(r["value"])
        abs_err = abs(val - ref_value) if ref_value is not None else None
        rel_err = (abs_err / abs(ref_value)) if ref_value is not None else None
        results.append(
            {
                "degree": d,
                "method_used": r["method_used"],
                "value": val,
                "error_estimate": r["error_estimate"],
                "abs_error": abs_err,
                "rel_error": rel_err,
                "base_node": r["base_node"],
            }
        )
    return {"decision": decision, "results": results}


def evaluate_pharmacokinetics(x_nodes, y_nodes, x_eval):
    """
    Evaluate drug concentration C(t) at x_eval using Lagrange interpolation
    and analyze minimum and peak concentrations when evaluated on a grid.

    Parameters
    ----------
    x_nodes : list or 1-D array
        Sampling times t (h).
    y_nodes : list or 1-D array
        Measured plasma concentrations C (mg/L).
    x_eval : float or 1-D numpy array
        Time(s) at which to evaluate the concentration interpolant.

    Returns
    -------
    dict
        Contains:
          "value"         : float or 1-D ndarray of concentration estimate(s)
          "degree"        : polynomial degree (n)
          "extrapolating" : bool
          "decision"      : decision dict for scalar x_eval (or midpoint of array)
          "min_val"       : float (minimum value across x_eval)
          "min_t"         : float (time of minimum value)
          "max_val"       : float (maximum value across x_eval, i.e., C_max)
          "max_t"         : float (time of maximum value, i.e., t_max)

    Raises
    ------
    TypeError
        If inputs are non-numeric.
    ValueError
        If nodes are repeated, empty, mismatched in length, or non-finite.

    Example
    -------
    >>> t = [0, 0.5, 1, 2]
    >>> c = [0.0, 4.215, 6.033, 6.516]
    >>> out = evaluate_pharmacokinetics(t, c, 1.5)
    >>> out["degree"]
    3
    """
    x_arr, y_arr = check_data(x_nodes, y_nodes)
    res = lagrange_interpolate(x_arr, y_arr, x_eval)
    vals = np.asarray(res["value"], dtype=float)
    x_eval_arr = np.asarray(x_eval, dtype=float)

    sample_x = float(x_eval_arr) if x_eval_arr.ndim == 0 else float(x_eval_arr[len(x_eval_arr) // 2])
    decision = choose_method(x_arr, y_arr, sample_x)

    if x_eval_arr.ndim == 0:
        min_val = float(vals)
        min_t = float(x_eval_arr)
        max_val = float(vals)
        max_t = float(x_eval_arr)
        value_out = float(vals)
    else:
        min_idx = int(np.argmin(vals))
        max_idx = int(np.argmax(vals))
        min_val = float(vals[min_idx])
        min_t = float(x_eval_arr[min_idx])
        max_val = float(vals[max_idx])
        max_t = float(x_eval_arr[max_idx])
        value_out = vals

    return {
        "value": value_out,
        "degree": res["degree"],
        "extrapolating": res["extrapolating"],
        "decision": decision,
        "min_val": min_val,
        "min_t": min_t,
        "max_val": max_val,
        "max_t": max_t,
    }


def evaluate_traffic_hermite(x_nodes, y_nodes, dydx_nodes, x_eval, speed_limit=100.0 / 3.6):
    """
    Check checkpoint speeds/average speeds and evaluate Hermite position H(t)
    and speed H'(t) at x_eval.

    Parameters
    ----------
    x_nodes : list or 1-D array
        Checkpoint times t (s).
    y_nodes : list or 1-D array
        Checkpoint positions s (m).
    dydx_nodes : list or 1-D array
        Instantaneous speeds v = s'(t) (m/s) at checkpoints.
    x_eval : float or 1-D numpy array
        Time(s) t at which to estimate position and speed.
    speed_limit : float, default 27.77777777777778 (100 km/h in m/s)
        Speed limit in m/s.

    Returns
    -------
    dict
        Contains:
          "position"           : float or 1-D ndarray, H(x_eval) in meters
          "speed"              : float or 1-D ndarray, H'(x_eval) in m/s
          "degree"             : int, degree of the Hermite polynomial (2n+1)
          "decision"           : dict from choose_method
          "avg_speeds"         : 1-D ndarray of average speeds on each interval
          "all_recorded_below" : bool, True if all dydx_nodes < speed_limit
          "all_avg_below"      : bool, True if all avg_speeds < speed_limit
          "exceeds_limit"      : bool, True if any H'(x_eval) > speed_limit
          "max_speed"          : float, maximum of H'(x_eval)
          "max_speed_time"     : float, time in x_eval where H'(t) is maximized

    Raises
    ------
    TypeError
        If inputs are non-numeric.
    ValueError
        If lengths mismatch, nodes repeat, or values contain nan/inf.

    Example
    -------
    >>> t = [0, 12, 25]
    >>> s = [0, 312, 655]
    >>> v = [24.5, 25.0, 24.0]
    >>> res = evaluate_traffic_hermite(t, s, v, 18.0)
    >>> res["degree"]
    5
    """
    x_arr, y_arr = check_data(x_nodes, y_nodes)
    v_arr = np.asarray(dydx_nodes, dtype=float)
    if len(v_arr) != len(x_arr):
        raise ValueError("dydx_nodes must have the same length as x_nodes.")

    dt = np.diff(x_arr)
    ds = np.diff(y_arr)
    avg_speeds = ds / dt

    all_recorded_below = bool(np.all(v_arr < speed_limit))
    all_avg_below = bool(np.all(avg_speeds < speed_limit))

    herm = hermite_interpolate(x_arr, y_arr, v_arr, x_eval, return_derivative=True)
    pos = np.asarray(herm["value"], dtype=float)
    spd = np.asarray(herm["derivative"], dtype=float)
    x_eval_arr = np.asarray(x_eval, dtype=float)

    sample_x = float(x_eval_arr) if x_eval_arr.ndim == 0 else float(x_eval_arr[0])
    decision = choose_method(x_arr, y_arr, sample_x, dydx_nodes=v_arr)

    if x_eval_arr.ndim == 0:
        max_spd = float(spd)
        max_spd_t = float(x_eval_arr)
        pos_out = float(pos)
        spd_out = float(spd)
    else:
        idx = int(np.argmax(spd))
        max_spd = float(spd[idx])
        max_spd_t = float(x_eval_arr[idx])
        pos_out = pos
        spd_out = spd

    return {
        "position": pos_out,
        "speed": spd_out,
        "degree": herm["degree"],
        "decision": decision,
        "avg_speeds": avg_speeds,
        "all_recorded_below": all_recorded_below,
        "all_avg_below": all_avg_below,
        "exceeds_limit": bool(np.any(spd > speed_limit)),
        "max_speed": max_spd,
        "max_speed_time": max_spd_t,
    }


def run_application_a(years, population, eval_years=(1955.0, 2015.0), degree=3):
    """
    Run Application A (Demography: U.S. Census Population) and print results.

    Parameters
    ----------
    years : list or 1-D array
        Census years.
    population : list or 1-D array
        Population in millions.
    eval_years : tuple of float, default (1955.0, 2015.0)
        Target years to estimate.
    degree : int, default 3
        Degree for Newton forward/backward interpolation.

    Returns
    -------
    dict
        Dictionary mapping each target year to its interpolation result dict.

    Raises
    ------
    ValueError
        If years and population lengths mismatch or are not equally spaced.

    Example
    -------
    >>> y = [1950, 1960, 1970, 1980]
    >>> p = [151.3, 179.3, 203.3, 226.5]
    >>> out = run_application_a(y, p, eval_years=(1955.0,), degree=2)  # doctest: +ELLIPSIS
    ===...
    >>> 1955.0 in out
    True
    """
    print("=" * 78)
    print("APPLICATION A: Population Between Census Years (Demography)")
    print("=" * 78)

    for yr in eval_years:
        dec = choose_method(years, population, float(yr))
        print(f"choose_method at x_eval = {yr:g}: '{dec['method']}'")
        print(f"  Reason: {dec['reason']}")
    print("  Agreement: Yes. The census years are equally spaced (h = 10), no derivatives")
    print("  are given, 1955 is near the beginning of the table (Newton forward), and")
    print("  2015 is near the end of the table (Newton backward).\n")

    print("--- Task (a): Forward Difference Table ---")
    print_difference_table(years, population, decimals=4)
    print("\nAnalysis of differences:")
    print("  Unlike e^x in Problem 2 (where higher differences decreased smoothly and")
    print("  remained positive), the census differences beyond Delta^2 y oscillate in")
    print("  sign and grow in magnitude (in the top row, Delta^3 y through Delta^7 y")
    print("  swing from +3.2 to -3.4 to +15.1 to -54.2 to +137.4). Real demographic data")
    print("  reflects baby booms, immigration shifts, and rounding noise rather than a")
    print("  single analytic curve. Therefore, a low degree (degree 2 or 3) using local")
    print("  nodes is sensible, whereas a high-degree global polynomial would amplify")
    print("  those large higher differences and oscillate.\n")

    print(f"--- Task (b): Population Estimates (degree {degree}) ---")
    results = {}
    for yr in eval_years:
        res = evaluate_census_interpolation(years, population, float(yr), degree=degree)
        results[float(yr)] = res
        param_name = "s" if "s" in res else "t"
        param_val = res.get("s", res.get("t"))
        est_str = "None" if res["error_estimate"] is None else f"{res['error_estimate']:.4f}"
        print(
            f"  Year {yr:g} ({res['method_used']}, degree={res['degree']}, "
            f"base_node={res['base_node']:g}, {param_name}={param_val:g}):"
        )
        print(f"    Estimated population = {res['value']:.4f} million")
        print(f"    Terms added          = {[round(float(t), 5) for t in res['terms']]}")
        print(f"    Error estimate       = {est_str} million\n")

    return results


def run_application_b(
    temps,
    pressures,
    eval_points=((25.0, 3.1698), (55.0, 15.763)),
    degrees=(1, 2, 3, 4),
    plot_path="steam_table.png",
):
    """
    Run Application B (Thermodynamics: Steam Table) and save the saturation
    pressure plot.

    Parameters
    ----------
    temps : list or 1-D array
        Temperatures T (deg C).
    pressures : list or 1-D array
        Saturation pressures P_sat (kPa).
    eval_points : tuple of (float, float)
        Pairs of (T_eval, P_ref) to evaluate and compare.
    degrees : tuple of int, default (1, 2, 3, 4)
        Degrees of interpolation to test.
    plot_path : str, default "steam_table.png"
        Output path for the saturation pressure figure.

    Returns
    -------
    dict
        Mapping from T_eval to the output of evaluate_steam_table.

    Raises
    ------
    ValueError
        If temps and pressures have mismatched lengths or unequal spacing.

    Example
    -------
    >>> T = [20, 30, 40, 50, 60]
    >>> P = [2.3392, 4.2469, 7.3851, 12.352, 19.947]
    >>> out = run_application_b(T, P, plot_path="steam_table.png")  # doctest: +ELLIPSIS
    ===...
    >>> 25.0 in out and 55.0 in out
    True
    """
    print("=" * 78)
    print("APPLICATION B: Reading Between the Lines of a Steam Table (Thermodynamics)")
    print("=" * 78)

    all_results = {}
    for t_eval, p_ref in eval_points:
        out = evaluate_steam_table(temps, pressures, t_eval, degrees=degrees, ref_value=p_ref)
        all_results[t_eval] = out
        dec = out["decision"]
        print(f"choose_method at T = {t_eval:g} deg C: '{dec['method']}'")
        print(f"  Reason: {dec['reason']}")
    print("  Agreement: Yes. Temperatures are equally spaced (h = 10 deg C) with no")
    print("  derivatives; 25 deg C lies in the first half (Newton forward) and 55 deg C")
    print("  lies in the second half (Newton backward).\n")

    print("--- Tasks (a) & (b): Estimates, Relative Errors, and Error Estimates ---")
    for t_eval, p_ref in eval_points:
        out = all_results[t_eval]
        print(f"\nT = {t_eval:g} deg C  (Reference P_sat = {p_ref:.4f} kPa, Method = {out['decision']['method']}):")
        print(
            f"{'Deg':>4} | {'Base T':>7} | {'P_sat (kPa)':>13} | {'Abs Error':>11} | "
            f"{'Rel Error (%)':>13} | {'error_estimate':>14}"
        )
        print("-" * 74)
        for r in out["results"]:
            est_str = "None" if r["error_estimate"] is None else f"{r['error_estimate']:.4e}"
            rel_pct = 100.0 * r["rel_error"]
            print(
                f"{r['degree']:>4} | {r['base_node']:>7.1f} | {r['value']:>13.6f} | "
                f"{r['abs_error']:>11.4e} | {rel_pct:>12.4f}% | {est_str:>14}"
            )

    # Task (c): Plot P_sat against T
    grid_t = np.linspace(min(temps), max(temps), 500)
    p_deg4 = evaluate_steam_table(temps, pressures, grid_t, degrees=(4,))["value"]

    plt.figure(figsize=(8, 5.5))
    plt.plot(grid_t, p_deg4, "b-", linewidth=2, label="Degree-4 Newton interpolant")
    plt.plot(
        temps,
        pressures,
        "k--",
        linewidth=1.2,
        alpha=0.7,
        label="Piecewise linear interpolation (chords)",
    )
    plt.plot(temps, pressures, "ro", markersize=7, label="Steam table data nodes")

    for t_eval, p_ref in eval_points:
        r1 = all_results[t_eval]["results"][0]["value"]
        r3 = all_results[t_eval]["results"][2]["value"]
        plt.plot(
            t_eval,
            r1,
            "ms",
            markersize=7,
            label=f"Linear est. at {t_eval:g} °C ({r1:.3f} kPa)" if t_eval == eval_points[0][0] else "_nolegend_",
        )
        plt.plot(
            t_eval,
            p_ref,
            "g^",
            markersize=8,
            label=f"Reference P_sat ({p_ref:.3f} kPa)" if t_eval == eval_points[0][0] else "_nolegend_",
        )

    plt.title("Saturation Pressure of Water $P_{sat}$ vs. Temperature $T$")
    plt.xlabel("Temperature $T$ (°C)")
    plt.ylabel("Saturation Pressure $P_{sat}$ (kPa)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n--- Task (c): Curve Shape & Engineering Safety (Plot saved to '{plot_path}') ---")
    print("  1. Why linear interpolation overestimates: By the Clausius-Clapeyron relation,")
    print("     saturation pressure grows roughly exponentially with T, so P_sat(T) is")
    print("     strictly convex (P''_sat(T) > 0). Any secant line (linear interpolant)")
    print("     connecting two points on a strictly convex curve lies above the curve.")
    print("     Equivalently, the quadratic error term s(s-1)/2 * Delta^2 P has s in (0,1),")
    print("     so s(s-1) < 0 while Delta^2 P > 0, meaning the linear term is too high.")
    print("  2. Safer error in pressure vessel design: When internal pressure determines")
    print("     required wall thickness (e.g., t = P*R / sigma), an OVERESTIMATE of pressure")
    print("     is the safer (conservative) error because it leads to a slightly thicker,")
    print("     stronger wall, whereas an underestimate could lead to an under-designed")
    print("     vessel that fails under operating pressure.")
    print("  3. Recommendation: Use degree 3 (4 terms) or degree 4 (5 terms). Going from")
    print("     degree 1 (2 terms) to degree 3 (4 terms) cuts the relative error from")
    print("     3.8883% (at 25 deg C) and 2.4519% (at 55 deg C) down to 0.2153% and")
    print("     0.0509%, while still leaving the 4th difference to provide an automatic")
    print("     error_estimate (degree 4 lowers relative error further to 0.0326% and 0.0011%).\n")

    return all_results


def run_application_c(
    times,
    concentrations,
    t_target=3.0,
    grid_points=2000,
    plot_path="pharmacokinetics.png",
):
    """
    Run Application C (Pharmacokinetics: Drug Concentration in the Blood)
    and save the pharmacokinetic curve plot.

    Parameters
    ----------
    times : list or 1-D array
        Sampling times t (h).
    concentrations : list or 1-D array
        Plasma drug concentrations C (mg/L).
    t_target : float, default 3.0
        Target time (h) at which to estimate concentration.
    grid_points : int, default 2000
        Number of evaluation points on [min(times), max(times)].
    plot_path : str, default "pharmacokinetics.png"
        Output path for the pharmacokinetic curve plot.

    Returns
    -------
    dict
        Summary results including C(3 h), global minimum, global peak, and local peak.

    Raises
    ------
    ValueError
        If times and concentrations have mismatched lengths or repeated nodes.

    Example
    -------
    >>> t = [0, 0.5, 1, 2, 4, 6, 8, 12]
    >>> c = [0.000, 4.215, 6.033, 6.516, 4.543, 2.809, 1.709, 0.629]
    >>> res = run_application_c(t, c, plot_path="pharmacokinetics.png")  # doctest: +ELLIPSIS
    ===...
    >>> res["c_at_target"] > 0
    True
    """
    print("=" * 78)
    print("APPLICATION C: Drug Concentration in the Blood (Pharmacokinetics)")
    print("=" * 78)

    pt_res = evaluate_pharmacokinetics(times, concentrations, t_target)
    dec = pt_res["decision"]
    print(f"choose_method at t = {t_target:g} h: '{dec['method']}'")
    print(f"  Reason: {dec['reason']}")
    print("  Agreement: Yes. The blood samples are drawn at irregular intervals")
    print("  (gaps of 0.5, 0.5, 1, 2, 2, 2, 4 h) and no derivatives are given,")
    print("  so Lagrange interpolation is required.\n")

    print("--- Task (a): Why Neither Newton Forward nor Backward Can Be Used ---")
    print("  Newton's forward- and backward-difference formulas assume uniform step size")
    print("  h = x_{i+1} - x_i across all nodes. Here the time steps vary from 0.5 h")
    print("  early on to 4.0 h later (max deviation from average h = 1.714 h is 2.286 h),")
    print("  so check_equal_spacing raises a ValueError and refuses to run.\n")

    print(f"--- Task (b): Degree-7 Lagrange Estimate at t = {t_target:g} h ---")
    print(f"  C({t_target:g} h) = {pt_res['value']:.6f} mg/L  (degree = {pt_res['degree']})\n")

    grid_t = np.linspace(min(times), max(times), grid_points)
    grid_res = evaluate_pharmacokinetics(times, concentrations, grid_t)
    c_grid = grid_res["value"]

    neg_mask = c_grid < 0
    neg_intervals = []
    if np.any(neg_mask):
        neg_t = grid_t[neg_mask]
        t_start, t_end = float(neg_t[0]), float(neg_t[-1])
        neg_intervals.append((t_start, t_end))

    # Also compute a local cubic Lagrange interpolant on the 4 nodes around the peak: [0.5, 1, 2, 4]
    local_t = np.array(times[1:5], dtype=float)
    local_c = np.array(concentrations[1:5], dtype=float)
    local_grid_t = np.linspace(local_t[0], local_t[-1], 2000)
    local_res = evaluate_pharmacokinetics(local_t, local_c, local_grid_t)
    local_at_3 = evaluate_pharmacokinetics(local_t, local_c, t_target)["value"]

    print(f"--- Task (c): Global Degree-7 Interpolant on [0, 12] ({grid_points} points) ---")
    print(f"  Predicts negative concentration? {bool(np.any(neg_mask))}")
    if neg_intervals:
        print(
            f"  Where negative: t in [{neg_intervals[0][0]:.4f} h, {neg_intervals[0][1]:.4f} h] "
            f"(between the nodes t = 8 h and t = 12 h)"
        )
    print(f"  Most negative value: C_min = {grid_res['min_val']:.4f} mg/L at t = {grid_res['min_t']:.4f} h")
    print("  Why physically impossible & lesson: A drug concentration cannot be negative")
    print("  (you cannot have less than zero drug mass per liter of blood). Fitting a single")
    print("  high-degree (degree 7) polynomial blindly through all 8 wide and unevenly")
    print("  spaced points triggers Runge-like oscillations in the wide gap [8, 12] h.")
    print("  Using a low-degree local interpolant (or spline/compartmental model) avoids")
    print("  spurious oscillations.\n")

    print("--- Task (d): Peak Concentration C_max and Time to Peak t_max ---")
    print(
        f"  From global degree-7 polynomial : C_max = {grid_res['max_val']:.4f} mg/L "
        f"at t_max = {grid_res['max_t']:.4f} h"
    )
    print(
        f"  From local degree-3 polynomial  : C_max = {local_res['max_val']:.4f} mg/L "
        f"at t_max = {local_res['max_t']:.4f} h  (nodes t = 0.5, 1, 2, 4 h)"
    )
    print(
        f"  (Note: The local cubic also estimates C(3 h) = {local_at_3:.4f} mg/L vs "
        f"global {pt_res['value']:.4f} mg/L, without any negative dip on [8, 12] h.)\n"
    )

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(grid_t, c_grid, "b-", linewidth=2, label="Global degree-7 Lagrange interpolant")
    plt.plot(
        local_grid_t,
        local_res["value"],
        "g--",
        linewidth=1.8,
        label="Local degree-3 Lagrange (t in [0.5, 4] h)",
    )
    plt.axhline(0, color="k", linestyle=":", linewidth=1)
    if np.any(neg_mask):
        plt.fill_between(
            grid_t,
            c_grid,
            0,
            where=neg_mask,
            color="salmon",
            alpha=0.4,
            label=f"Unphysical negative region (min = {grid_res['min_val']:.2f} mg/L)",
        )
    plt.plot(times, concentrations, "ro", markersize=7, label="Measured blood samples")
    plt.plot(
        t_target,
        pt_res["value"],
        "ms",
        markersize=7,
        label=f"C(3 h) = {pt_res['value']:.3f} mg/L",
    )
    plt.plot(
        grid_res["max_t"],
        grid_res["max_val"],
        "k^",
        markersize=8,
        label=f"Peak C_max = {grid_res['max_val']:.3f} mg/L at t_max = {grid_res['max_t']:.2f} h",
    )
    plt.title("Pharmacokinetic Curve: Plasma Drug Concentration $C(t)$ vs. Time $t$")
    plt.xlabel("Time $t$ (h)")
    plt.ylabel("Plasma Concentration $C$ (mg/L)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to '{plot_path}'.\n")

    return {
        "c_at_target": pt_res["value"],
        "global_min_val": grid_res["min_val"],
        "global_min_t": grid_res["min_t"],
        "neg_interval": neg_intervals[0] if neg_intervals else None,
        "global_c_max": grid_res["max_val"],
        "global_t_max": grid_res["max_t"],
        "local_c_max": local_res["max_val"],
        "local_t_max": local_res["max_t"],
    }


def run_application_d(
    times,
    positions,
    speeds,
    eval_times=(18.0, 33.0, 46.0),
    speed_limit_kmh=100.0,
    plot_path="speed_profile.png",
):
    """
    Run Application D (Traffic Enforcement: Hermite Interpolation) and save
    the speed profile plot.

    Parameters
    ----------
    times : list or 1-D array
        Checkpoint times t (s).
    positions : list or 1-D array
        Checkpoint positions s (m).
    speeds : list or 1-D array
        Checkpoint instantaneous speeds v = s'(t) (m/s).
    eval_times : tuple of float, default (18.0, 33.0, 46.0)
        Times (s) at which to estimate position and speed.
    speed_limit_kmh : float, default 100.0
        Posted speed limit in km/h.
    plot_path : str, default "speed_profile.png"
        Output path for the speed profile figure.

    Returns
    -------
    dict
        Summary results including estimates at eval_times and maximum speed on [0, 52].

    Raises
    ------
    ValueError
        If times, positions, and speeds do not have matching lengths.

    Example
    -------
    >>> t = [0, 12, 25, 40, 52]
    >>> s = [0, 312, 655, 1062, 1378]
    >>> v = [24.5, 25.0, 24.0, 25.5, 24.0]
    >>> res = run_application_d(t, s, v, plot_path="speed_profile.png")  # doctest: +ELLIPSIS
    ===...
    >>> res["max_speed"] > 27.78
    True
    """
    print("=" * 78)
    print("APPLICATION D: Was the Car Speeding? (Traffic Enforcement — Hermite)")
    print("=" * 78)

    speed_limit_ms = speed_limit_kmh / 3.6
    grid_t = np.linspace(min(times), max(times), 2001)
    grid_res = evaluate_traffic_hermite(times, positions, speeds, grid_t, speed_limit=speed_limit_ms)
    dec = grid_res["decision"]

    print(f"choose_method: '{dec['method']}'")
    print(f"  Reason: {dec['reason']}")
    print("  Agreement: Yes. Both position s(t_i) and its derivative v(t_i) = s'(t_i) are")
    print("  measured at every checkpoint, so Hermite interpolation is the only method")
    print("  that honors both position and instantaneous speed constraints.\n")

    print(
        f"--- Task (a): Recorded & Average Speeds vs. Limit "
        f"({speed_limit_kmh:g} km/h = {speed_limit_ms:.4f} m/s) ---"
    )
    print(f"  Recorded instantaneous speeds (m/s): {list(speeds)}")
    print(f"  Max recorded speed                 : {max(speeds):.2f} m/s ({max(speeds)*3.6:.2f} km/h) < {speed_limit_ms:.2f} m/s")
    print("\n  Interval average speeds v_avg = Delta s / Delta t:")
    for i in range(len(times) - 1):
        v_avg = grid_res["avg_speeds"][i]
        print(
            f"    [{times[i]:>2g} s, {times[i+1]:>2g} s]: "
            f"Delta s = {positions[i+1] - positions[i]:>4g} m, "
            f"Delta t = {times[i+1] - times[i]:>2g} s  ->  "
            f"v_avg = {v_avg:.4f} m/s ({v_avg * 3.6:.2f} km/h)"
        )
    print(f"  All recorded speeds below limit? {grid_res['all_recorded_below']}")
    print(f"  All average speeds below limit?  {grid_res['all_avg_below']}")
    print("  Conclusion from raw numbers alone: No, based ONLY on checkpoint speeds and")
    print("  average speeds between checkpoints, every value is <= 27.1333 m/s (97.68 km/h),")
    print("  so the driver would NOT be fined by a standard spot-speed or average-speed camera.\n")

    print("--- Task (b): Which Interpolation Method This Data Calls For and Why ---")
    print("  Method: Hermite interpolation (degree 2n+1 = 9 for 5 nodes).")
    print("  Why: We are given both the function value s(t_i) and its first derivative")
    print("  v(t_i) = s'(t_i) at 5 unequally spaced checkpoints. Lagrange or Newton")
    print("  would ignore the speedometer readings v(t_i) (and Newton cannot handle")
    print("  the unequal time steps 12, 13, 15, 12 s). Hermite matches both position")
    print("  and velocity at every checkpoint.\n")

    print(f"--- Task (c): Hermite (degree {grid_res['degree']}) Estimates at t = 18, 33, 46 s ---")
    speed_label = "Speed H'(t) (m/s)"
    print(
        f"{'t (s)':>7} | {'Position H(t) (m)':>18} | {speed_label:>18} | "
        f"{'Speed (km/h)':>14} | {'Exceeds Limit?':>15}"
    )
    print("-" * 80)
    pt_estimates = {}
    for te in eval_times:
        r = evaluate_traffic_hermite(times, positions, speeds, float(te), speed_limit=speed_limit_ms)
        pt_estimates[float(te)] = r
        spd_kmh = r["speed"] * 3.6
        exceeds = "YES (SPEEDING)" if r["speed"] > speed_limit_ms else "No"
        print(
            f"{te:>7.1f} | {r['position']:>18.4f} | {r['speed']:>18.4f} | "
            f"{spd_kmh:>14.2f} | {exceeds:>15}"
        )

    over_mask = grid_res["speed"] > speed_limit_ms
    over_t = grid_t[over_mask]
    # Find contiguous intervals where speed > speed_limit_ms
    speeding_intervals = []
    if np.any(over_mask):
        padded = np.concatenate(([False], over_mask, [False]))
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0] - 1
        for s_idx, e_idx in zip(starts, ends):
            speeding_intervals.append((float(grid_t[s_idx]), float(grid_t[e_idx])))

    print(
        f"\n  Peak reconstructed speed on [0, 52] s: H'_max = {grid_res['max_speed']:.4f} m/s "
        f"({grid_res['max_speed'] * 3.6:.2f} km/h) at t = {grid_res['max_speed_time']:.2f} s."
    )
    if speeding_intervals:
        iv_str = ", ".join(f"[{a:.2f} s, {b:.2f} s]" for a, b in speeding_intervals)
        print(f"  Intervals where H'(t) > {speed_limit_ms:.2f} m/s: {iv_str}")
    print("  Why the car must have sped between checkpoints: In every interval, the")
    print("  recorded speeds at both checkpoints (24.0 to 25.5 m/s) are noticeably lower")
    print("  than the average speed across the interval (26.00 to 27.13 m/s). That means")
    print("  the driver slowed down for each camera and accelerated between cameras.")
    print("  In particular, on [25, 40] s the average speed is 27.1333 m/s while the")
    print("  endpoint speeds are 24.0 and 25.5 m/s, forcing H'(t) above the 27.78 m/s")
    print("  limit (reaching 28.97 m/s = 104.29 km/h at t = 33 s, and peaking at 29.59 m/s")
    print("  on [25, 40] s and 32.60 m/s near t = 50 s in the degree-9 polynomial).\n")

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(grid_t, grid_res["speed"], "b-", linewidth=2, label="Hermite speed profile $H'(t)$ (degree 9)")
    plt.axhline(
        speed_limit_ms,
        color="r",
        linestyle="--",
        linewidth=1.8,
        label=f"Speed limit ({speed_limit_kmh:g} km/h ≈ {speed_limit_ms:.2f} m/s)",
    )
    if np.any(over_mask):
        plt.fill_between(
            grid_t,
            grid_res["speed"],
            speed_limit_ms,
            where=over_mask,
            color="salmon",
            alpha=0.45,
            label="Speeding zone ($H'(t) > 27.78$ m/s)",
        )
    plt.plot(times, speeds, "ko", markersize=7, label="Checkpoint speed readings $v(t_i)$")

    eval_t_arr = np.array(eval_times, dtype=float)
    eval_v_arr = np.array([pt_estimates[float(t)]["speed"] for t in eval_times], dtype=float)
    plt.plot(eval_t_arr, eval_v_arr, "ms", markersize=7, label="Evaluated speeds at t = 18, 33, 46 s")

    plt.title("Reconstructed Vehicle Speed Profile $H'(t)$ on $[0, 52]$ s")
    plt.xlabel("Time $t$ (s)")
    plt.ylabel("Speed $v = H'(t)$ (m/s)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to '{plot_path}'.\n")

    return {
        "estimates": pt_estimates,
        "max_speed": grid_res["max_speed"],
        "max_speed_time": grid_res["max_speed_time"],
        "speeding_interval": (float(over_t[0]), float(over_t[-1])) if len(over_t) > 0 else None,
    }


if __name__ == "__main__":
    # Application A: Population Between Census Years (Demography)
    census_years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], dtype=float)
    census_pop = np.array([151.3, 179.3, 203.3, 226.5, 248.7, 281.4, 308.7, 331.4], dtype=float)
    run_application_a(census_years, census_pop, eval_years=(1955.0, 2015.0), degree=3)

    # Application B: Reading Between the Lines of a Steam Table (Thermodynamics)
    steam_T = np.array([20, 30, 40, 50, 60], dtype=float)
    steam_Psat = np.array([2.3392, 4.2469, 7.3851, 12.352, 19.947], dtype=float)
    run_application_b(
        steam_T,
        steam_Psat,
        eval_points=((25.0, 3.1698), (55.0, 15.763)),
        degrees=(1, 2, 3, 4),
        plot_path="steam_table.png",
    )

    # Application C: Drug Concentration in the Blood (Pharmacokinetics)
    pk_t = np.array([0, 0.5, 1, 2, 4, 6, 8, 12], dtype=float)
    pk_C = np.array([0.000, 4.215, 6.033, 6.516, 4.543, 2.809, 1.709, 0.629], dtype=float)
    run_application_c(pk_t, pk_C, t_target=3.0, grid_points=2000, plot_path="pharmacokinetics.png")

    # Application D: Was the Car Speeding? (Traffic Enforcement — Hermite)
    car_t = np.array([0, 12, 25, 40, 52], dtype=float)
    car_s = np.array([0, 312, 655, 1062, 1378], dtype=float)
    car_v = np.array([24.5, 25.0, 24.0, 25.5, 24.0], dtype=float)
    run_application_d(
        car_t,
        car_s,
        car_v,
        eval_times=(18.0, 33.0, 46.0),
        speed_limit_kmh=100.0,
        plot_path="speed_profile.png",
    )
