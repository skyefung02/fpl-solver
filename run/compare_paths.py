# Written by Skye Fung, 2025
# Path comparison tool with optional matched-seed robustness analysis.
# Deterministic mode: one solve per path, results grouped by path with headline summary.
# Robustness mode (N_RUNS > 0): N matched-seed randomised solves per path; paths are
#   compared by win rate — how often each path scores highest under the same noise draw.
#   Using matched seeds controls for draw difficulty, isolating the effect of the locked player.
# All paths run in parallel (one process per job); seeds are independent across processes.

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from solve import print_transfer_chip_summary, solve_regular

from utils import cached_request, load_settings


# ─── USER CONFIGURATION ───────────────────────────────────────────────────────

# Settings applied to every path, overriding user_settings.json.
SOLVER_OPTIONS = {
    "solver": "gurobi",
    "verbose": False,
    "print_result_table": False,
    "print_decay_metrics": False,
    "print_transfer_chip_summary": False,
    "print_squads": False,
    "parallel": "off",
    "num_iterations": 1,
    # "gap": 0.002,          # uncomment to trade accuracy for speed (0.2% optimality gap)
}

# Player IDs to force as sells this GW across ALL paths.
# The GW number is fetched automatically from the FPL API — no need to hardcode it.
# Leave as [] if you don't want to constrain the sells.
#
# Find a player's FPL ID by searching your projection CSV:
#   grep -i "playername" data/projection_all_metrics.csv | head -1
FORCED_SELLS = [
]

# Paths to compare. Each entry is a dict with:
#   "name"           — label shown in output (required)
#   "locked_next_gw" — list of player IDs forced into squad this GW (optional)
#   Any other solver option to override for this specific path only.
PATHS = [
    # {
    #     "name": "WC32, BB33, FH34",
    # },
    # {
    #     "name": "FH29 (WC+BB free)",
    #     "use_fh": [29],
    #     "use_wc": [],    # clears the WC32 from user_settings.json
    #     "use_bb": [],    # clears the BB33 from user_settings.json
    #     "chip_limits": {"wc": 1, "bb": 1, "fh": 1, "tc": 0},
    # },
    # {
    #     "name": "Ekitike",
    #     "locked_next_gw": [661],
    #     "num_transfers": 1
    # },
    # {
    #     "name": "Salah + Ekitike",
    #     "locked_next_gw": [381, 661],
    # },
    {
        "name": "Salah + DCL",
        "locked_next_gw": [381, 691],
    },
    {
        "name": "Salah + Watkins",
        "locked_next_gw": [381, 64],
    },
    # Add more paths below:
    # {
    #     "name": "Schade",
    #     "locked_next_gw": [???],   # fill in Schade's FPL ID
    # },
]

# True  → subprocess output hidden, only progress bar shown (recommended)
# False → full solver output printed per path (useful for debugging)
SUPPRESS_OUTPUT = True

# N_RUNS controls which mode runs:
#   N_RUNS = 1  → deterministic path comparison (one solve per path, base projections)
#   N_RUNS > 1  → matched-seed robustness analysis (N randomised solves per path)
#                 Each path shares the same seeds, so draw difficulty is identical —
#                 win rate (how often a path scores highest) is the ranking metric.
#                 Robustness solves use horizon=6 and gap=0.002 for speed.
N_RUNS = 1
RANDOMIZATION_STRENGTH = 0.9

# ─── END USER CONFIGURATION ───────────────────────────────────────────────────


def _solve_silent(args):
    """Redirect stdout to /dev/null at OS level to suppress HiGHS C++ output.
    Same approach as run_parallel.py — verbose=False alone is insufficient
    because HiGHS writes directly to the OS file descriptor, bypassing Python's
    sys.stdout. The fd is restored in the finally block so the subprocess
    stays healthy even if solve_regular raises an exception."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        return solve_regular(args)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def _solve_silent_with_response(args):
    """Like _solve_silent but returns (result_table, response) for horizon printing."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        return solve_regular({**args, "_return_response": True})
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def _solve_regular_with_response(args):
    """Like solve_regular but returns (result_table, response) for horizon printing."""
    return solve_regular({**args, "_return_response": True})


def _get_next_gw():
    """Get the next gameweek from the FPL API."""
    fpl_data = cached_request("https://fantasy.premierleague.com/api/bootstrap-static/")
    for event in fpl_data["events"]:
        if event["is_next"]:
            return event["id"]
    return None


def _filter_chips_to_horizon(settings, horizon, next_gw):
    """Drop chip GWs that fall outside [next_gw, next_gw + horizon - 1].

    Mirrors the same function in run_parallel.py — needed whenever the robustness
    solves use a shorter horizon than user_settings.json may have assumed.
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


def _signal_strength(p, n):
    """Classify win-rate signal using the lower bound of the 95% confidence interval.

    A signal is 'Noise' if its CI lower bound touches zero — i.e. the observed
    win rate is statistically indistinguishable from zero at N runs.
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


def _print_path_horizons(path_results_full):
    """Print GW-by-GW transfer/chip plan for each path (best iteration, ranked by score)."""
    sorted_results = sorted(path_results_full, key=lambda x: -float(x[1].iloc[0]["score"]))

    print(f"\n{'=' * 66}")
    print(f"  Full Horizon Plans (best iteration per path, ranked by score)")
    print(f"{'=' * 66}")

    for name, df, response in sorted_results:
        best_iter = int(df.iloc[0]["iter"])
        best_result = next(r for r in response if r["iter"] == best_iter)
        score = df.iloc[0]["score"]
        total_xp = sum(gw_stats.get("xP", 0) for _, gw_stats in best_result["statistics"].items())

        print(f"\n--- {name} | Score: {score:.2f} | Total xP: {total_xp:.2f} ---")
        print_transfer_chip_summary(best_result, {})


def _print_path_comparison(path_results, next_gw):
    """Print a headline summary table followed by full iterations grouped by path."""
    gw_label = f"GW{next_gw}" if next_gw else "GW?"
    n_iters = max(len(df) for _, df in path_results)

    print(f"\n{'=' * 66}")
    print(f"  Path Comparison — {gw_label}, {n_iters} iterations per path")
    print(f"{'=' * 66}")

    # --- Headline: best iter per path, sorted by score ---
    headline_rows = []
    for name, df in path_results:
        best = df.iloc[0]
        sell = str(best["sell"]).strip()
        buy = str(best["buy"]).strip()
        headline_rows.append([
            name,
            sell if sell not in ("-", "", "nan") else "Roll",
            buy if buy not in ("-", "", "nan") else "Roll",
            best.get("chip", ""),
            f"{best['score']:.2f}",
        ])

    headline_rows.sort(key=lambda x: -float(x[4]))

    print("\nHEADLINE — best iter per path, ranked by score")
    print(tabulate(
        headline_rows,
        headers=["Path", "Sell", "Buy", "Chip", "Score"],
        tablefmt="simple",
    ))

    # --- Full iterations per path ---
    print("\n\nFULL ITERATIONS BY PATH")
    for name, df in path_results:
        print(f"\n--- {name} ---")
        display = df[["iter", "sell", "buy", "chip", "score"]].copy()
        print(tabulate(display, headers="keys", tablefmt="simple", showindex=False, floatfmt=".2f"))

    print()


def run_path_comparison(paths, solver_options, forced_sells=None, suppress_output=True):
    """Run one solve per path (with num_iterations each) and print grouped results.

    Paths are run in parallel (one subprocess per path). Iterations within each
    path are sequential — solve_regular handles them internally.

    Args:
        paths: list of path dicts (see USER CONFIGURATION above)
        solver_options: dict of options applied to every path
        forced_sells: list of player IDs to force as transfer_out this GW across all paths.
                      The GW number is resolved automatically from the FPL API.
        suppress_output: if True, suppress HiGHS stdout from subprocesses
    """
    worker_fn = _solve_silent_with_response if suppress_output else _solve_regular_with_response
    next_gw = _get_next_gw()

    # Build booked_transfers entries for forced sells, using the live next_gw
    forced_sell_entries = []
    if forced_sells and next_gw:
        forced_sell_entries = [{"gw": next_gw, "transfer_out": pid} for pid in forced_sells]
        print(f"[forced sells] GW{next_gw}: {forced_sells}")

    # Build merged runtime options per path
    run_args = []
    for path in paths:
        name = path.get("name", "Unnamed")
        path_overrides = {k: v for k, v in path.items() if k != "name"}
        merged = {**solver_options, **path_overrides}
        # Merge forced sells into any existing booked_transfers for this path
        if forced_sell_entries:
            existing = merged.get("booked_transfers", [])
            merged["booked_transfers"] = existing + forced_sell_entries
        run_args.append((name, merged))

    max_workers = max(1, os.cpu_count() - 2)
    path_results = [None] * len(run_args)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_fn, args): (i, name)
            for i, (name, args) in enumerate(run_args)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Comparing paths", unit="path"):
            i, name = futures[future]
            df, response = future.result()
            df = df.reset_index(drop=True)
            path_results[i] = (name, df, response)

    _print_path_horizons(path_results)
    _print_path_comparison([(name, df) for name, df, _ in path_results], next_gw)

    # Save full results with path label
    all_rows = []
    for name, df, _ in path_results:
        df = df.copy()
        df.insert(0, "path", name)
        all_rows.append(df)

    out_path = "path_comparison.csv"
    pd.concat(all_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Full results saved to {out_path}")


def run_robustness_comparison(paths, solver_options, n_runs, randomization_strength, forced_sells=None, suppress_output=True):
    """Run matched-seed robustness comparison across paths.

    For each of n_runs seeds, all paths are solved with identical noise draws.
    The primary metric is win rate: how often each path achieves the highest
    score across all paths for a given seed. Using matched seeds controls for
    draw difficulty — a hard draw hurts all paths equally, so the winner is
    determined solely by the relative advantage of each locked player.

    Robustness solves use horizon=6 and gap=0.002 for speed. Results are saved
    to path_robustness.csv.

    Args:
        paths: list of path dicts (see USER CONFIGURATION)
        solver_options: base options applied to every solve
        n_runs: number of seeds (total solves = n_runs × len(paths))
        randomization_strength: noise scale passed to the solver
        forced_sells: list of player IDs to force as transfer_out this GW
        suppress_output: if True, suppress HiGHS stdout from subprocesses
    """
    worker_fn = _solve_silent if suppress_output else solve_regular
    next_gw = _get_next_gw()
    gw_label = f"GW{next_gw}" if next_gw else "GW?"

    forced_sell_entries = []
    if forced_sells and next_gw:
        forced_sell_entries = [{"gw": next_gw, "transfer_out": pid} for pid in forced_sells]

    # Robustness-specific options: shorter horizon and relaxed gap for speed.
    # Apply chip filter so chips set beyond horizon=6 don't cause empty-group warnings.
    _horizon = 6
    base_settings = load_settings()
    chip_overrides = _filter_chips_to_horizon(base_settings, _horizon, next_gw)
    robustness_opts = {
        **solver_options,
        **chip_overrides,
        "horizon": _horizon,
        "gap": 0.002,
        "randomized": True,
        "randomization_strength": randomization_strength,
    }

    # Build one job per (seed, path) combination.
    path_names = [p.get("name", "Unnamed") for p in paths]
    jobs = []  # list of (name, seed, args)
    for seed in range(n_runs):
        for path in paths:
            name = path.get("name", "Unnamed")
            path_overrides = {k: v for k, v in path.items() if k != "name"}
            merged = {**robustness_opts, **path_overrides, "randomization_seed": seed}
            if forced_sell_entries:
                existing = merged.get("booked_transfers", [])
                merged["booked_transfers"] = existing + forced_sell_entries
            jobs.append((name, seed, merged))

    # Run all jobs in parallel.
    max_workers = max(1, os.cpu_count() - 2)
    results = {}  # (name, seed) -> score

    total = len(jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_fn, args): (name, seed)
            for name, seed, args in jobs
        }
        for future in tqdm(as_completed(futures), total=total, desc="Robustness", unit="solve"):
            name, seed = futures[future]
            df = future.result().reset_index(drop=True)
            results[(name, seed)] = df.iloc[0]["score"]

    # For each seed, award a win to the highest-scoring path.
    # Ties split the win fractionally so total wins always sum to n_runs.
    win_counts = {name: 0.0 for name in path_names}
    for seed in range(n_runs):
        seed_scores = {name: results[(name, seed)] for name in path_names}
        best_score = max(seed_scores.values())
        winners = [name for name, score in seed_scores.items() if score == best_score]
        for winner in winners:
            win_counts[winner] += 1.0 / len(winners)

    # Compute per-path score statistics across all seeds.
    path_stats = {}
    for name in path_names:
        scores = sorted(results[(name, seed)] for seed in range(n_runs))
        path_stats[name] = {
            "wins": win_counts[name],
            "median": scores[len(scores) // 2],
            "min": scores[0],
            "max": scores[-1],
        }

    # Print summary table.
    print(f"\n{'=' * 70}")
    print(f"  Robustness Comparison — {gw_label}, N={n_runs} seeds, horizon=6")
    print(f"{'=' * 70}")
    print(f"  Win rate: fraction of seeds where this path scores highest.")
    print(f"  All paths share the same seeds — draw difficulty is controlled.")

    rows = []
    for name in path_names:
        s = path_stats[name]
        wins = s["wins"]
        p = wins / n_runs
        rows.append([
            name,
            f"{wins:.0f}",
            f"{100 * p:.0f}%",
            f"{s['median']:.1f}",
            f"{s['min']:.1f}",
            f"{s['max']:.1f}",
            _signal_strength(p, n_runs),
        ])
    rows.sort(key=lambda x: -float(x[1]))

    print()
    print(tabulate(rows, headers=["Path", "Wins", "Win%", "Median", "Min", "Max", "Signal"], tablefmt="simple"))
    print()

    # Save per-seed scores to CSV.
    rob_rows = [
        {"path": name, "seed": seed, "score": results[(name, seed)]}
        for name in path_names
        for seed in range(n_runs)
    ]
    out_path = "path_robustness.csv"
    pd.DataFrame(rob_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Robustness results saved to {out_path}")


if __name__ == "__main__":
    if N_RUNS <= 1:
        run_path_comparison(PATHS, SOLVER_OPTIONS, forced_sells=FORCED_SELLS, suppress_output=SUPPRESS_OUTPUT)
    else:
        run_robustness_comparison(
            PATHS, SOLVER_OPTIONS, N_RUNS, RANDOMIZATION_STRENGTH,
            forced_sells=FORCED_SELLS, suppress_output=SUPPRESS_OUTPUT,
        )
