# Substantially modified by Skye Fung, 2025
# Original work: fploptimised.com, licensed under Apache 2.0
# Changes: _signal_strength() CI-based signal classification;
#          _print_summary() robustness summary table;
#          _get_next_gw() FPL API GW fetch;
#          _filter_chips_to_horizon() auto chip horizon filter;
#          _solve_silent() OS-level stdout suppression via os.dup2;
#          run_parallel_solves() chip filter, tqdm progress bar,
#            as_completed pattern, parallel="off" for macOS, gap=0.002;
#          __main__ block switched to randomised stress test (N=50,
#            randomization_strength=1.2)

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from solve import solve_regular

from utils import cached_request, get_dict_combinations, load_settings


def _signal_strength(p, n):
    """Classify signal strength using the lower bound of the 95% confidence interval.

    A signal is 'Noise' if its CI lower bound touches zero — i.e. the observed
    frequency is statistically indistinguishable from zero at N runs.
    Thresholds for N=50: Noise <~8%, Weak 8-20%, Moderate 20-30%, Strong 30-45%, Very Strong 45%+
    """
    if p == 0:
        return "—"
    lower = p - 1.96 * (p * (1 - p) / n) ** 0.5
    if lower <= 0:
        return "Noise"
    elif lower < 0.10:
        return "Weak"
    elif lower < 0.20:
        return "Moderate"
    elif lower < 0.35:
        return "Strong"
    else:
        return "Very Strong"


def _print_summary(df, n_runs, next_gw, horizon):
    """Print a robustness summary instead of the raw 50-row results table.

    Shows two tables:
    - ACTION FREQUENCY: how often each (sell → buy) pair appears, ranked by count.
      'Roll' means no transfer was made. Includes signal strength classification.
    - PLAYER FREQUENCY: how often each individual player appears as a sell or buy
      target, aggregated across all action combinations.

    Signal strength is based on the lower bound of the 95% CI: if the lower bound
    is above zero, the signal is statistically distinguishable from noise at N runs.
    Full results are still saved to chip_solve.csv for reference.
    """
    gw_label = f"GW{next_gw}" if next_gw else "GW?"

    print(f"\n{'=' * 62}")
    print(f"  Robustness Analysis — {gw_label}, horizon={horizon}, N={n_runs}")
    print(f"  Score: {df['score'].min():.1f} – {df['score'].max():.1f}  |  Median: {df['score'].median():.1f}")
    print(f"{'=' * 62}")

    # --- Action frequency (sell → buy pairs) ---
    df = df.copy()

    def _is_empty(val):
        return str(val).strip() in ("-", "", "nan")

    df["action"] = df.apply(
        lambda r: "Roll" if _is_empty(r["sell"]) and _is_empty(r["buy"]) else f"{r['sell']} → {r['buy']}",
        axis=1,
    )
    action_counts = df["action"].value_counts()
    action_rows = []
    for action, count in action_counts.items():
        p = count / n_runs
        action_rows.append([action, count, f"{100 * p:.0f}%", _signal_strength(p, n_runs)])

    print("\nACTION FREQUENCY")
    print(tabulate(action_rows, headers=["Action", "Count", "%", "Signal"], tablefmt="simple"))

    # --- Individual player frequency ---
    def _parse_names(cell):
        s = str(cell).strip()
        return [] if s in ("-", "", "nan") else [name.strip() for name in s.split(",")]

    sell_counter = Counter(name for names in df["sell"].map(_parse_names) for name in names)
    buy_counter = Counter(name for names in df["buy"].map(_parse_names) for name in names)

    player_rows = []
    for player, count in sell_counter.items():
        p = count / n_runs
        player_rows.append([player, "SELL", count, f"{100 * p:.0f}%", _signal_strength(p, n_runs)])
    for player, count in buy_counter.items():
        p = count / n_runs
        player_rows.append([player, "BUY", count, f"{100 * p:.0f}%", _signal_strength(p, n_runs)])

    player_rows.sort(key=lambda x: -x[2])

    if player_rows:
        print("\nPLAYER FREQUENCY")
        print(tabulate(player_rows, headers=["Player", "Dir", "Count", "%", "Signal"], tablefmt="simple"))

    print()


def _get_next_gw(settings):
    """Get the next gameweek from a settings override or the FPL API."""
    if settings.get("override_next_gw"):
        return int(settings["override_next_gw"])
    fpl_data = cached_request("https://fantasy.premierleague.com/api/bootstrap-static/")
    for event in fpl_data["events"]:
        if event["is_next"]:
            return event["id"]
    return None


def _filter_chips_to_horizon(settings, horizon, next_gw):
    """Return chip override dict with out-of-horizon GWs removed.

    Reads use_wc/use_bb/use_fh/use_tc from user_settings.json (via settings),
    then drops any GW that falls outside [next_gw, next_gw + horizon - 1].
    Prints a one-line warning listing everything that was dropped.
    """
    if next_gw is None:
        return {}
    last_gw = next_gw + horizon - 1
    overrides = {}
    dropped = []
    for chip in ["use_wc", "use_bb", "use_fh", "use_tc"]:
        gws_for_chip = settings.get(chip, [])
        in_horizon = [gw for gw in gws_for_chip if next_gw <= gw <= last_gw]
        out_of_horizon = [gw for gw in gws_for_chip if gw < next_gw or gw > last_gw]
        if out_of_horizon:
            dropped.append(f"{chip[4:].upper()} GW{out_of_horizon}")
        overrides[chip] = in_horizon
    if dropped:
        print(f"[chip filter] Dropped chip(s) outside horizon (GW{next_gw}–GW{last_gw}): {', '.join(dropped)}")
    return overrides


def _solve_silent(args):
    """Wrapper around solve_regular that suppresses all stdout from the subprocess.
    This keeps the terminal clean when running many solves in parallel — only the
    progress bar printed by the main process will be visible.
    Set SUPPRESS_SUBPROCESS_OUTPUT = False below to disable this and see full output.

    redirect_stdout() alone is insufficient because HiGHS is a C++ library that
    writes directly to the OS-level file descriptor (fd 1), bypassing Python's
    sys.stdout entirely. os.dup2 redirects at the OS level so C extensions are
    silenced too. The fd is restored in the finally block so the subprocess stays
    healthy even if solve_regular raises an exception."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        return solve_regular(args)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def run_parallel_solves(chip_combinations, max_workers=None, suppress_output=True):
    """Run multiple solves in parallel and print a ranked results table.

    Args:
        chip_combinations: list of option dicts, one per solve
        max_workers: number of parallel processes (defaults to cpu_count - 2)
        suppress_output: if True, subprocess solver output is hidden and only
                         Progress lines are shown. If False, all subprocess output
                         is printed (will be interleaved and messy with multiple workers).
    """
    if not max_workers:
        max_workers = os.cpu_count() - 2

    # These options are merged into every solve's settings.
    # - verbose=False: suppresses HiGHS solver progress table
    # - print_* flags: suppress the summary tables printed after each solve
    # - parallel="off": disables HiGHS's internal threading so multiple solves
    #   can run simultaneously without competing for CPU cores (important on macOS)
    # horizon - this is the gameweek horizon that we are solving for, in each solve 
    # gap - solver stops when it can prove the current best solution is within 0.2% of optimal
    options = {
        "verbose": False,
        "print_result_table": False,
        "print_decay_metrics": False,
        "print_transfer_chip_summary": False,
        "print_squads": False,
        "parallel": "off",
        "horizon": 6,
        "gap": 0.002,
        "num_iterations": 1
    }

    # Auto-filter chip constraints outside the parallel solver's shorter horizon.
    # user_settings.json may pin chips to GWs beyond horizon=6, which causes
    # "Requested variable group is empty" warnings. We read the base settings,
    # determine next_gw from the FPL API (cached), and strip out-of-range GWs.
    base_settings = load_settings()
    next_gw = _get_next_gw(base_settings)
    options.update(_filter_chips_to_horizon(base_settings, options["horizon"], next_gw))

    args = []
    for combination in chip_combinations:
        args.append({**options, **combination})

    # Choose worker function based on suppress_output flag.
    # _solve_silent redirects stdout to /dev/null inside each subprocess,
    # suppressing the data-loading prints that verbose=False doesn't cover
    # (e.g. "Filtered player pool", "Added player X", license notice, "OC -" lines).
    worker_fn = _solve_silent if suppress_output else solve_regular

    total = len(args)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, arg): i for i, arg in enumerate(args)}
        for future in tqdm(as_completed(futures), total=total, desc="Solving", unit="solve"):
            results.append(future.result())

    df = pd.concat(results).sort_values(by="score", ascending=False).reset_index(drop=True)
    df = df.drop("iter", axis=1)
    _print_summary(df, total, next_gw, options["horizon"])

    df.to_csv("chip_solve.csv", encoding="utf-8", index=False)


if __name__ == "__main__":
    # --- Output suppression toggle ---
    # True  → only "Progress: X/Y solves complete" lines are shown (recommended for parallel runs)
    # False → full solver output from every subprocess is printed (messy but useful for debugging)
    SUPPRESS_SUBPROCESS_OUTPUT = True

    # --- Randomized stress test ---
    # Runs N solves with different random noise applied to projections.
    # Each seed produces a different noise draw, so the solver sees slightly
    # different xP values each time. Players that appear as transfers across
    # most runs are robust picks; those that appear rarely are marginal.
    # Results are ranked by score and saved to chip_solve.csv.
    N_RUNS = 50
    scenarios = [{"randomized": True, "randomization_seed": i, "randomization_strength": 1.2} for i in range(N_RUNS)]
    run_parallel_solves(scenarios, suppress_output=SUPPRESS_SUBPROCESS_OUTPUT)

    # --- Chip comparison (commented out) ---
    # Uncomment below and comment out the randomized block above to compare chip strategies instead.
    # chip_gameweeks = {
    #     "use_bb": [None, 1, 2],
    #     "use_wc": [],
    #     "use_fh": [None, 2, 3, 4],
    #     "use_tc": [],
    # }
    # combinations = get_dict_combinations(chip_gameweeks)
    # run_parallel_solves(combinations, suppress_output=SUPPRESS_SUBPROCESS_OUTPUT)
