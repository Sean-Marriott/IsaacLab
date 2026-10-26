# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identify the damping of a passive hinge from a hang-and-release recording.

The payload is pulled away from rest, let go, and its joint angle is logged while the swing dies
out. From that decay this script estimates, per joint:

* the viscous damping ratio ``zeta`` (exponential decay, what a zero-stiffness drive models),
* the Coulomb friction dead band (linear decay, what PhysX joint friction models),
* the share of the energy loss that is Coulomb rather than viscous, which says which model fits,
* the damped period and natural frequency, which check the mass and inertia of the sim model,
* the equilibrium the swing oscillates about and the angle it finally settles at.

Both damping mechanisms are fitted together through the peak-to-peak recurrence
``A[k+1] = a * A[k] - b`` on successive half-cycle amplitudes ``A``. Pure viscous damping gives
``b = 0`` and ``a = exp(-pi * zeta / sqrt(1 - zeta^2))``; pure Coulomb friction gives ``a = 1`` and
``b`` equal to twice the dead-band half-width.

Input is a CSV with a header row, one time column in seconds and one column per joint angle. The
recording should cover a few seconds at rest, the hold, the release and the full decay. Each file
should contain a single release; pass several files to aggregate repeats.

Usage:

.. code-block:: bash

    # analyse every angle column of one recording
    ./isaaclab.sh -p scripts/tools/analyze_hinge_decay.py release_pitch_01.csv

    # aggregate five repeats of the pitch test, and save diagnostic plots
    ./isaaclab.sh -p scripts/tools/analyze_hinge_decay.py release_pitch_*.csv \\
        --cols csTubePitch --plot plots/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

parser = argparse.ArgumentParser(description="Identify passive hinge damping from hang-and-release recordings.")
parser.add_argument("csv_files", nargs="+", help="CSV recordings, one release per file.")
parser.add_argument("--time_col", type=str, default=None, help="Time column [s]. Defaults to the first column.")
parser.add_argument("--cols", nargs="+", default=None, help="Angle columns to analyse. Defaults to all others.")
parser.add_argument("--units", choices=["deg", "rad"], default="deg", help="Units of the angle columns.")
parser.add_argument("--start", type=float, default=None, help="Ignore samples before this time [s].")
parser.add_argument("--end", type=float, default=None, help="Ignore samples after this time [s].")
parser.add_argument(
    "--smooth", type=float, default=0.03, help="Width of the centred moving-average filter [s]. 0 disables it."
)
parser.add_argument(
    "--settle_window", type=float, default=2.0, help="Final stretch of the recording treated as settled [s]."
)
parser.add_argument(
    "--min_amp",
    type=float,
    default=0.2,
    help="Smallest half-cycle amplitude used in the fits, in the angle units. Raised to 5x the noise if lower.",
)
parser.add_argument("--plot", type=str, default=None, help="Directory to write one diagnostic PNG per file and joint.")
parser.add_argument("--json", type=str, default=None, help="Write all per-file results to this JSON file.")
args_cli = parser.parse_args()


@dataclass
class DecayResult:
    """Identified decay of one joint in one recording. Angles are in the input units."""

    file: str
    joint: str
    ok: bool
    message: str = ""
    num_half_cycles: int = 0
    initial_amplitude: float = math.nan
    zeta: float = math.nan
    """Viscous damping ratio from the combined fit."""
    zeta_viscous_only: float = math.nan
    """Damping ratio if all the loss is assumed viscous (log-linear fit)."""
    coulomb_dead_band: float = math.nan
    """Half-width of the Coulomb dead band from the combined fit."""
    coulomb_share: float = math.nan
    """Fraction of the per-half-cycle loss that is Coulomb, evaluated at the median amplitude."""
    rms_viscous_only: float = math.nan
    rms_coulomb_only: float = math.nan
    rms_combined: float = math.nan
    damped_period: float = math.nan
    """Damped period [s]."""
    natural_frequency: float = math.nan
    """Undamped natural angular frequency [rad/s]."""
    equilibrium: float = math.nan
    """Angle the swing oscillates about, from the peaks."""
    settled_angle: float = math.nan
    """Median angle over the settle window."""
    noise_std: float = math.nan
    warnings: list[str] = field(default_factory=list)


def load_csv(path: str) -> dict[str, np.ndarray]:
    """Read a numeric CSV with a header row into named columns, dropping rows with missing values."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = []
        for row in reader:
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                continue
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(header):
        raise ValueError(f"{path}: could not parse a numeric table matching the header {header}.")
    data = data[np.all(np.isfinite(data), axis=1)]
    return {name: data[:, i] for i, name in enumerate(header)}


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average that keeps the signal length, shrinking the window at the edges."""
    if width <= 1:
        return x.copy()
    kernel = np.ones(width)
    return np.convolve(x, kernel, mode="same") / np.convolve(np.ones_like(x), kernel, mode="same")


def find_half_cycle_peaks(t: np.ndarray, dev: np.ndarray, noise: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate the extremum of every half cycle of a decaying oscillation about zero.

    A crossing only counts once the signal has left a noise band on the other side, so chatter
    around zero does not split a half cycle in two.

    Returns:
        The peak indices, the crossing indices that bound them, and the crossing times [s].
    """
    # Hysteresis state: +1 / -1 once the deviation has left the band on that side, 0 before that.
    band = 3.0 * noise
    state = np.zeros_like(dev, dtype=int)
    current = 0
    for i, v in enumerate(dev):
        if v > band:
            current = 1
        elif v < -band:
            current = -1
        state[i] = current
    # At each confirmed side change, place the crossing at the last raw sign change before it.
    cross_idx, cross_t = [], []
    for i in np.nonzero((state[1:] != state[:-1]) & (state[:-1] != 0))[0] + 1:
        old = state[i - 1]
        j = i - 1
        while j > 0 and dev[j] * old <= 0:
            j -= 1
        cross_idx.append(j)
        cross_t.append(t[j] - dev[j] * (t[j + 1] - t[j]) / (dev[j + 1] - dev[j]))
    cross_idx_arr = np.asarray(cross_idx, dtype=int)
    cross_t_arr = np.asarray(cross_t)
    peaks = []
    bounds = np.concatenate(([0], cross_idx_arr, [len(dev) - 1]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo < 2:
            continue
        seg = dev[lo : hi + 1]
        peaks.append(lo + int(np.argmax(np.abs(seg))))
    return np.asarray(peaks, dtype=int), cross_idx_arr, cross_t_arr


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares ``y = slope * x + intercept``."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def zeta_from_ratio(ratio: float) -> float:
    """Damping ratio from the half-cycle amplitude ratio ``A[k+1] / A[k]`` of a viscous decay."""
    if not 0.0 < ratio < 1.0:
        return math.nan
    delta_half = -math.log(ratio)  # logarithmic decrement per half cycle
    return delta_half / math.sqrt(math.pi**2 + delta_half**2)


def refine_peak_values(t: np.ndarray, raw: np.ndarray, peaks: np.ndarray, hold_value: float) -> np.ndarray:
    """Estimate each peak's angle from a parabola fitted to the raw samples around it.

    Taking the largest sample biases small peaks upward by the noise, which makes a viscous decay
    look slower than it is. A parabola over a few dozen samples averages the noise out instead.
    The first peak is the release itself, where the payload leaves a flat hold rather than turning
    smoothly, so it is replaced by ``hold_value``, the mean angle over the end of the hold.
    """
    values = raw[peaks].astype(float).copy()
    if len(peaks) < 2:
        return values
    half_period = float(np.median(np.diff(t[peaks])))
    w = 0.16 * half_period  # +/- ~30 deg of phase, where a cosine is still close to a parabola
    for n, i in enumerate(peaks):
        win = np.abs(t - t[i]) <= w
        if n == 0:
            values[n] = hold_value
            continue
        if win.sum() < 5:
            continue
        tc = t[win] - t[i]
        c2, c1, c0 = np.polyfit(tc, raw[win], 2)
        if c2 == 0.0:
            continue
        tv = -c1 / (2.0 * c2)
        if abs(tv) <= w:
            values[n] = c0 - c1**2 / (4.0 * c2)
    return values


def leading_run(mask: np.ndarray) -> np.ndarray:
    """Indices of the first contiguous run of True values in ``mask``."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return idx
    gaps = np.nonzero(np.diff(idx) != 1)[0]
    return idx[: gaps[0] + 1] if len(gaps) else idx


def fit_recurrence(amps: np.ndarray) -> tuple[float, float, float]:
    """Fit ``A[k+1] = a * A[k] - b`` with ``0 < a <= 1`` and ``b >= 0``.

    Returns:
        The viscous ratio ``a``, the Coulomb drop ``b`` per half cycle, and the RMS residual.
    """
    x, y = amps[:-1], amps[1:]
    a, neg_b = fit_line(x, y)
    b = -neg_b
    if a >= 1.0:  # no detectable viscous part; pure Coulomb
        a, b = 1.0, float(np.mean(x - y))
    if b < 0.0:  # no detectable Coulomb part; pure viscous, through the origin
        a, b = float(np.dot(x, y) / np.dot(x, x)), 0.0
    resid = float(np.sqrt(np.mean((y - (a * x - b)) ** 2)))
    return a, b, resid


def fit_equilibrium(values: np.ndarray, guess: float) -> float:
    """Find the angle about which the peaks decay most consistently with :func:`fit_recurrence`.

    A wrong centre makes the half-cycle amplitudes alternate large-small, which neither damping
    mechanism produces, so the residual has a clear minimum at the true equilibrium. This holds for
    any mix of viscous and Coulomb loss, unlike closed-form three-peak estimates, which are exact
    for one mechanism only. With fewer than five peaks the centre is not identifiable alongside the
    two damping parameters, and the guess is returned unchanged.
    """
    if len(values) < 5:
        return guess
    span = 0.25 * float(np.max(np.abs(values - guess)))
    signs = np.sign(values - guess)

    def cost(c: float) -> float:
        if np.any(np.sign(values - c) != signs):
            return math.inf
        return fit_recurrence(np.abs(values - c))[2]

    best = guess
    for half_width, n in ((span, 401), (span / 100.0, 201)):
        grid = np.linspace(best - half_width, best + half_width, n)
        costs = [cost(c) for c in grid]
        best = float(grid[int(np.argmin(costs))])
    return best


def analyse(path: str, joint: str, t: np.ndarray, angle: np.ndarray) -> tuple[DecayResult, dict]:
    """Identify the decay of one joint. Returns the result and the arrays used for plotting."""
    res = DecayResult(file=os.path.basename(path), joint=joint, ok=False)
    dt = float(np.median(np.diff(t)))
    x = moving_average(angle, max(int(round(args_cli.smooth / dt)), 1)) if args_cli.smooth > 0 else angle.copy()
    plot = {"t": t, "x": x}

    # -- settled angle and noise floor from the final window
    settle = t >= t[-1] - args_cli.settle_window
    if settle.sum() < 10:
        res.message = "settle window holds fewer than 10 samples; record longer after the swing stops"
        return res, plot
    res.settled_angle = float(np.median(angle[settle]))
    res.noise_std = max(float(np.std(angle[settle])), 1e-9)
    min_amp = max(args_cli.min_amp, 5.0 * res.noise_std)

    # -- release: the last sample of the hold, where the deviation is still at its maximum
    dev = x - res.settled_angle
    peak_dev = float(np.max(np.abs(dev)))
    if peak_dev < 3.0 * min_amp:
        res.message = f"no clear swing (max deviation {peak_dev:.3g} vs threshold {min_amp:.3g})"
        return res, plot
    first_big = int(np.argmax(np.abs(dev) >= 0.95 * peak_dev))
    sign0 = np.sign(dev[first_big])
    leaves = np.nonzero(sign0 * dev[first_big:] < 0.95 * peak_dev)[0]
    release = first_big + (int(leaves[0]) - 1 if len(leaves) else 0)
    release = max(release, 0)
    plot["release_t"] = t[release]
    hold = (t >= t[release] - 0.2) & (t <= t[release])
    hold_value = float(np.mean(angle[hold]))

    # -- peaks about the settled angle first, then about the equilibrium that best fits the decay.
    # The settled angle is only the equilibrium without friction: Coulomb friction can stop the
    # swing anywhere inside its dead band, and every amplitude is measured from the equilibrium.
    tt, xx, raw = t[release:], x[release:], angle[release:]
    centre = res.settled_angle
    for _ in range(2):
        peaks, cross_idx, cross_t = find_half_cycle_peaks(tt, xx - centre, res.noise_std)
        values = refine_peak_values(tt, raw, peaks, hold_value)
        keep = leading_run(np.abs(values - centre) >= min_amp)
        peaks, values = peaks[keep], values[keep]
        if len(peaks) < 3:
            break
        centre = fit_equilibrium(values, centre)
    res.equilibrium = centre
    amps = np.abs(values - centre)
    plot.update({"peak_t": tt[peaks], "peak_x": values, "equilibrium": centre})
    res.num_half_cycles = len(amps)
    if len(amps) < 3:
        res.message = f"only {len(amps)} half cycles above {min_amp:.3g}; release from a larger angle"
        return res, plot
    res.initial_amplitude = float(amps[0])
    k = np.arange(len(amps), dtype=float)

    # -- period from the crossings that fall within the analysed peaks
    in_range = (cross_t > tt[peaks[0]]) & (cross_t < tt[peaks[-1]])
    ct = cross_t[in_range]
    if len(ct) >= 2:
        res.damped_period = float(2.0 * np.mean(np.diff(ct)))
    else:
        res.damped_period = float(2.0 * np.mean(np.diff(tt[peaks])))

    # -- pure viscous: ln A linear in half-cycle count
    s_v, c_v = fit_line(k, np.log(amps))
    pred_v = np.exp(c_v + s_v * k)
    res.zeta_viscous_only = zeta_from_ratio(math.exp(s_v))
    res.rms_viscous_only = float(np.sqrt(np.mean((amps - pred_v) ** 2)))

    # -- pure Coulomb: A linear in half-cycle count, dropping by twice the dead band each half cycle
    s_c, c_c = fit_line(k, amps)
    pred_c = c_c + s_c * k
    res.rms_coulomb_only = float(np.sqrt(np.mean((amps - pred_c) ** 2)))

    # -- combined: A[k+1] = a A[k] - b, simulated forward from the first peak for the residual
    a, b, _ = fit_recurrence(amps)
    pred = [amps[0]]
    for _ in range(len(amps) - 1):
        pred.append(a * pred[-1] - b)
    pred_comb = np.asarray(pred)
    res.rms_combined = float(np.sqrt(np.mean((amps - pred_comb) ** 2)))
    res.zeta = zeta_from_ratio(a) if a < 1.0 else 0.0
    res.coulomb_dead_band = b / 2.0
    a_med = float(np.median(amps))
    loss_v, loss_c = (1.0 - a) * a_med, b
    res.coulomb_share = loss_c / (loss_v + loss_c) if loss_v + loss_c > 0 else math.nan
    zeta_for_wn = res.zeta if math.isfinite(res.zeta) else 0.0
    res.natural_frequency = 2.0 * math.pi / res.damped_period / math.sqrt(max(1.0 - zeta_for_wn**2, 1e-6))
    plot.update({"k": k, "amps": amps, "pred_v": pred_v, "pred_c": pred_c, "pred_comb": pred_comb})

    # -- sanity checks
    if np.any(amps[1:] > amps[:-1] + max(3.0 * res.noise_std, 0.02 * amps[0])):
        res.warnings.append("amplitude grows between half cycles: likely coupling with another mode (beating)")
    if abs(res.settled_angle - res.equilibrium) > res.coulomb_dead_band + 3.0 * res.noise_std + 0.01 * amps[0]:
        res.warnings.append("settled angle lies outside the dead band about the equilibrium: check for drift")
    if len(amps) < 5:
        res.warnings.append("fewer than 5 half cycles: the decay type is poorly determined")
    res.ok = True
    return res, plot


def save_plot(res: DecayResult, plot: dict, out_dir: str) -> None:
    """Write the time series with the detected peaks, and the amplitude decay with the three fits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.plot(plot["t"], plot["x"], lw=0.8, label="angle (smoothed)")
    if "equilibrium" in plot:
        ax0.axhline(plot["equilibrium"], color="k", ls="--", lw=0.8, label="equilibrium")
    if "peak_t" in plot:
        ax0.plot(plot["peak_t"], plot["peak_x"], "o", ms=4, label="half-cycle peaks")
    if "release_t" in plot:
        ax0.axvline(plot["release_t"], color="r", lw=0.8, label="release")
    ax0.set_xlabel("time [s]")
    ax0.set_ylabel(f"angle [{args_cli.units}]")
    ax0.set_title(f"{res.file}: {res.joint}")
    ax0.legend(fontsize=8)
    if "amps" in plot:
        ax1.semilogy(plot["k"], plot["amps"], "o", label="measured")
        ax1.semilogy(plot["k"], plot["pred_v"], label=f"viscous only (zeta={res.zeta_viscous_only:.3f})")
        ax1.semilogy(plot["k"], np.clip(plot["pred_c"], 1e-9, None), label="Coulomb only")
        ax1.semilogy(
            plot["k"],
            np.clip(plot["pred_comb"], 1e-9, None),
            "--",
            label=f"combined (zeta={res.zeta:.3f}, band={res.coulomb_dead_band:.3g})",
        )
        ax1.set_xlabel("half cycle")
        ax1.set_ylabel(f"amplitude [{args_cli.units}]")
        ax1.set_title("straight = viscous, bending down = Coulomb")
        ax1.legend(fontsize=8)
    fig.tight_layout()
    name = f"{os.path.splitext(res.file)[0]}_{res.joint}.png".replace(os.sep, "_")
    fig.savefig(os.path.join(out_dir, name), dpi=120)
    plt.close(fig)


def verdict(share: float) -> str:
    """Name the dominant damping mechanism from the Coulomb share of the loss."""
    if not math.isfinite(share):
        return "?"
    if share < 0.25:
        return "viscous"
    if share > 0.75:
        return "Coulomb"
    return "mixed"


def main():
    """Analyse every requested joint in every file and print per-file and aggregated results."""
    u = args_cli.units
    results: list[DecayResult] = []
    for path in args_cli.csv_files:
        cols = load_csv(path)
        names = list(cols)
        time_col = args_cli.time_col or names[0]
        if time_col not in cols:
            sys.exit(f"{path}: time column '{time_col}' not found in {names}")
        t = cols[time_col]
        order = np.argsort(t)
        mask = np.ones_like(t, dtype=bool)
        if args_cli.start is not None:
            mask &= t >= args_cli.start
        if args_cli.end is not None:
            mask &= t <= args_cli.end
        order = order[mask[order]]
        joints = args_cli.cols or [n for n in names if n != time_col]
        for joint in joints:
            if joint not in cols:
                sys.exit(f"{path}: angle column '{joint}' not found in {names}")
            res, plot = analyse(path, joint, t[order], cols[joint][order])
            results.append(res)
            if args_cli.plot:
                save_plot(res, plot, args_cli.plot)

    print(f"\nPer recording (angles in {u}):")
    header = (
        f"{'file':<28} {'joint':<14} {'n':>3} {'A0':>7} {'zeta':>7} {'zeta_v':>7} {'band':>7} "
        f"{'coul%':>6} {'T_d[s]':>7} {'wn':>7} {'equil':>8} {'settled':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if not r.ok:
            print(f"{r.file:<28} {r.joint:<14} skipped: {r.message}")
            continue
        print(
            f"{r.file:<28} {r.joint:<14} {r.num_half_cycles:>3} {r.initial_amplitude:>7.2f} {r.zeta:>7.4f}"
            f" {r.zeta_viscous_only:>7.4f} {r.coulomb_dead_band:>7.3f} {100 * r.coulomb_share:>5.0f}%"
            f" {r.damped_period:>7.3f} {r.natural_frequency:>7.3f} {r.equilibrium:>8.2f} {r.settled_angle:>8.2f}"
            f"  {verdict(r.coulomb_share)}"
        )
        for w in r.warnings:
            print(f"{'':<44}! {w}")

    print("\nAcross recordings (mean +/- std):")
    for joint in dict.fromkeys(r.joint for r in results):
        ok = [r for r in results if r.joint == joint and r.ok]
        if not ok:
            print(f"  {joint}: no usable recordings")
            continue

        def stat(attr: str) -> str:
            v = np.array([getattr(r, attr) for r in ok], dtype=float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                return "n/a"
            return f"{v.mean():.4g} +/- {v.std(ddof=1) if len(v) > 1 else 0.0:.2g}"

        share = float(np.nanmean([r.coulomb_share for r in ok]))
        print(f"  {joint} ({len(ok)} recordings, decay looks {verdict(share)})")
        print(f"    zeta (combined)      {stat('zeta')}")
        print(f"    zeta (viscous only)  {stat('zeta_viscous_only')}")
        print(f"    Coulomb dead band    {stat('coulomb_dead_band')} {u}")
        print(f"    Coulomb share        {stat('coulomb_share')}")
        print(f"    damped period        {stat('damped_period')} s")
        print(f"    natural frequency    {stat('natural_frequency')} rad/s")
        print(f"    equilibrium          {stat('equilibrium')} {u}")
        print(f"    settled angle        {stat('settled_angle')} {u}")
        # Amplitude dependence separates the mechanisms too: viscous zeta is constant, while a Coulomb
        # loss makes the apparent viscous-only zeta grow as the release angle shrinks.
        if len(ok) >= 3:
            a0 = np.array([r.initial_amplitude for r in ok])
            zv = np.array([r.zeta_viscous_only for r in ok])
            if np.ptp(a0) > 0.3 * a0.mean() and np.all(np.isfinite(zv)):
                slope = np.polyfit(a0, zv, 1)[0]
                change = slope * np.ptp(a0) / zv.mean()
                if abs(change) < 0.2:
                    trend = "is about constant across release angles (viscous signature)"
                elif slope < 0:
                    trend = f"falls {-100 * change:.0f}% over the release angles (Coulomb signature)"
                else:
                    trend = f"rises {100 * change:.0f}% over the release angles (check for mode coupling)"
                print(f"    viscous-only zeta {trend}")

    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nWrote {args_cli.json}")


if __name__ == "__main__":
    main()
