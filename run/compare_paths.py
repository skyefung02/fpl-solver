# Written by Skye Fung, 2025
# Deterministic path comparison tool.
# Runs one solve per defined path, each with num_iterations alternatives.
# Results are displayed grouped by path with a headline summary for quick comparison.
# Runs paths in parallel (one process per path); iterations within each path are sequential.

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from solve import solve_regular

from utils import cached_request, load_settings


# ─── USER CONFIGURATION ───────────────────────────────────────────────────────

# Settings applied to every path, overriding user_settings.json.
SOLVER_OPTIONS = {
    "verbose": False,
    "print_result_table": False,
    "print_decay_metrics": False,
    "print_transfer_chip_summary": False,
    "print_squads": False,
    "parallel": "off",       # critical on macOS — prevents HiGHS competing with subprocess workers
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
    237,   # Enzo
    256,   # Muñoz
]

# Paths to compare. Each entry is a dict with:
#   "name"           — label shown in output (required)
#   "locked_next_gw" — list of player IDs forced into squad this GW (optional)
#   Any other solver option to override for this specific path only.
PATHS = [
    {
        "name": "Free",
    },
    {
        "name": "Ndiaye",
        "locked_next_gw": [299],
    },
    {
        "name": "Wilson",
        "locked_next_gw": [329],
    },
    {
        "name": "Schade",
        "locked_next_gw": [120],
    },
    {
        "name": "Dango",
        "locked_next_gw": [83],
    },
    {
        "name": "Wirtz",
        "locked_next_gw": [382],
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


def _get_next_gw():
    """Get the next gameweek from the FPL API."""
    fpl_data = cached_request("https://fantasy.premierleague.com/api/bootstrap-static/")
    for event in fpl_data["events"]:
        if event["is_next"]:
            return event["id"]
    return None


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
    worker_fn = _solve_silent if suppress_output else solve_regular
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
            df = future.result().reset_index(drop=True)
            path_results[i] = (name, df)

    _print_path_comparison(path_results, next_gw)

    # Save full results with path label
    all_rows = []
    for name, df in path_results:
        df = df.copy()
        df.insert(0, "path", name)
        all_rows.append(df)

    out_path = "path_comparison.csv"
    pd.concat(all_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    run_path_comparison(PATHS, SOLVER_OPTIONS, forced_sells=FORCED_SELLS, suppress_output=SUPPRESS_OUTPUT)
