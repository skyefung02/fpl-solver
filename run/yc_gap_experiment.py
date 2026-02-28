#!/usr/bin/env python3
"""
Yellow-card gap experiment: DEF and MID, GW29-38.

For every DEF/MID player-GW observation with xMins > 60 (SGW only),
run simulate() and compute:
    gap = sim_mean - file_ev

If Solio prices in yellow-card risk but the simulator does not:
    E[yellow pts] = -1 × p_yc × P(played | xmins)
    => gap ≈ p_yc × P(played | xmins)

Columns in report:
  Name, Pos, Team, GW, xMins, file_ev, sim_mean, gap, implied_p_yc_per90
  where implied_p_yc_per90 = gap / (xmins/90)  -- the per-game rate needed
  to explain the observed overestimation purely via yellow cards.

Usage:
    python run/yc_gap_experiment.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paths import DATA_DIR
from run.monte_carlo_sim import (
    DEFAULT_PEN_CONV,
    load_player_data,
    simulate,
)

N_SIMS    = 100_000
SEED      = 42
GWS       = list(range(29, 39))
POSITIONS = ("D", "M")
MIN_XMINS = 60.0   # only confident starters
DGW_THRESH = 95.0  # xMins > this → DGW, excluded


def main() -> None:
    proj = pd.read_csv(DATA_DIR / "projection_all_metrics.csv")
    pen  = pd.read_csv(DATA_DIR / "penalties.csv")
    fix  = pd.read_csv(DATA_DIR / "fixture_difficulty_all_metrics.csv")
    rng  = np.random.default_rng(SEED)

    records = []

    for pos_filter in POSITIONS:
        sub = proj[proj["Pos"] == pos_filter].copy()

        for gw in GWS:
            xmins_col = f"{gw}_xMins"
            pts_col   = f"{gw}_Pts"
            if xmins_col not in sub.columns:
                continue

            eligible = sub[
                (sub[xmins_col] > MIN_XMINS) & (sub[xmins_col] <= DGW_THRESH)
            ]

            for _, row in eligible.iterrows():
                name = row["Name"]
                team = row["Team"]

                try:
                    p = load_player_data(
                        proj, pen, name, gw, DEFAULT_PEN_CONV,
                        fix=fix, team=team,
                    )
                except ValueError:
                    continue

                sim_arr  = simulate(p, N_SIMS, rng)
                sim_mean = float(sim_arr.mean())
                file_ev  = p["ev"]
                gap      = sim_mean - file_ev  # positive → simulator overestimates

                # Implied p_yc per 90 mins needed to explain the gap entirely via
                # yellow cards (each worth -1 pt).
                # E[yc_pts] = -p_yc × P(played | xmins)  ≈  -p_yc × (xmins/90)
                # => gap = p_yc × (xmins/90)  [gap positive when overestimating]
                implied_p_yc = gap / (p["xmins"] / 90.0) if p["xmins"] > 0 else np.nan

                records.append({
                    "Name":              name,
                    "Pos":               pos_filter,
                    "Team":              team,
                    "GW":                gw,
                    "xMins":             p["xmins"],
                    "file_ev":           round(file_ev, 3),
                    "sim_mean":          round(sim_mean, 3),
                    "gap":               round(gap, 3),
                    "implied_p_yc_p90":  round(implied_p_yc, 4),
                })

    df = pd.DataFrame(records)

    # ── Per-position summary ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Gap summary by position  (sim_mean − file_ev)")
    print("  SGW only, xMins > 60,  GW29–38")
    print("═" * 60)
    for pos, grp in df.groupby("Pos"):
        g = grp["gap"]
        pct_pos = (g > 0).mean() * 100
        print(f"\n  Position: {pos}  ({len(grp)} player-GW observations)")
        print(f"    Mean gap          : {g.mean():+.4f}")
        print(f"    Median gap        : {g.median():+.4f}")
        print(f"    Std gap           : {g.std():.4f}")
        print(f"    % overestimating  : {pct_pos:.1f}%")
        print(f"    Implied p_yc/90   : {grp['implied_p_yc_p90'].mean():.4f}  "
              f"(mean across {len(grp)} obs)")

    # ── Per-player aggregate ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Per-player aggregated gap  (sorted by mean gap desc)")
    print("═" * 60)
    for pos, grp in df.groupby("Pos"):
        agg = (
            grp.groupby(["Name", "Team"])
            .agg(
                n_obs=("gap", "count"),
                mean_gap=("gap", "mean"),
                mean_p_yc=("implied_p_yc_p90", "mean"),
                total_file_ev=("file_ev", "sum"),
                total_sim_mean=("sim_mean", "sum"),
            )
            .reset_index()
            .sort_values("mean_gap", ascending=False)
        )

        print(f"\n  {pos}  — top 20 by mean gap")
        print(f"  {'Name':<22} {'Team':<20} {'n':>3}  {'mean_gap':>9}  {'impl_p_yc/90':>12}")
        print(f"  {'─'*22} {'─'*20} {'─'*3}  {'─'*9}  {'─'*12}")
        for _, r in agg.head(20).iterrows():
            print(f"  {r['Name']:<22} {r['Team']:<20} {int(r['n_obs']):>3}  "
                  f"{r['mean_gap']:>+9.3f}  {r['mean_p_yc']:>12.4f}")

        print(f"\n  {pos}  — bottom 10 by mean gap  (least overestimated / underestimated)")
        print(f"  {'Name':<22} {'Team':<20} {'n':>3}  {'mean_gap':>9}  {'impl_p_yc/90':>12}")
        print(f"  {'─'*22} {'─'*20} {'─'*3}  {'─'*9}  {'─'*12}")
        for _, r in agg.tail(10).iterrows():
            print(f"  {r['Name']:<22} {r['Team']:<20} {int(r['n_obs']):>3}  "
                  f"{r['mean_gap']:>+9.3f}  {r['mean_p_yc']:>12.4f}")

    # ── Save full results ───────────────────────────────────────────────────────
    out = ROOT / "tmp" / "yc_gap_experiment.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Full results saved → {out}")
    print(f"  Total observations: {len(df)}")


if __name__ == "__main__":
    main()
