#!/usr/bin/env python3
"""
Monte Carlo simulator for comparing two FPL players with similar EV.

Decomposes each player's expected points into independent scoring events
(minutes, np goals, assists, clean sheet, bonus, CBIT, penalties) and
samples 100k+ scenarios to produce a full points distribution.

Penalty handling:
  - penalties.csv holds P(player takes a penalty) per GW.
  - The existing `goals` column already bakes in the expected penalty goal
    contribution, so the simulator backs out the non-penalty goal rate:
      λ_np = max(0, goals − p_pen × PENALTY_CONVERSION)
  - Each simulation then draws: pen_taken ~ Bernoulli(p_pen × mins/90),
    pen_scored ~ Bernoulli(PENALTY_CONVERSION).
  - Outcome: +goal_pts if scored, −2 if taken but missed.

Usage:
    python run/monte_carlo_sim.py --gw 28 --player1 "Gordon" --player2 "Rice"
    python run/monte_carlo_sim.py --gw 28 --player1 "Gordon" --player2 "Rice" --n_sims 200000 --no_plot
    python run/monte_carlo_sim.py --gw 28 --player1 "Haaland" --player2 "M.Salah" --pen_conv 0.78
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paths import DATA_DIR  # noqa: E402

# ── FPL scoring rules ──────────────────────────────────────────────────────────
GOAL_PTS          = {"G": 6, "D": 6, "M": 5, "F": 4}
CS_PTS            = {"G": 6, "D": 6, "M": 1, "F": 0}
ASSIST_PTS        = 3
CBIT_PTS          = 2       # 2-point bonus for exceeding CBIT threshold
PENALTY_MISS_PTS  = -2      # points deducted for a missed penalty
MINS_FULL         = 60      # minutes threshold for full appearance bonus
DEFAULT_PEN_CONV  = 0.76    # Premier League average penalty conversion rate
# ──────────────────────────────────────────────────────────────────────────────


def load_player_data(
    proj: pd.DataFrame,
    pen: pd.DataFrame,
    name: str,
    gw: int,
    pen_conv: float,
) -> dict:
    """Extract per-GW projection and penalty stats for a named player.

    Both projection_all_metrics.csv and penalties.csv come from Solio, but
    Solio uses player-specific (unknown) conversion rates when building the
    goals column, so there is no reliable universal rate to back-calculate
    non-penalty goal lambdas.

    Instead, 28_goals is used directly as the Poisson rate for total goals
    (already reflecting Solio's own penalty contribution).  The penalties.csv
    data is used only to model the penalty miss risk (−2 pts), which is NOT
    captured in the 28_Pts projection.
    """
    proj_matches = proj[proj["Name"] == name]
    if proj_matches.empty:
        raise ValueError(f"Player '{name}' not found. Check spelling / capitalisation.")
    row = proj_matches.iloc[0]
    g   = str(gw)

    # Penalty probability — default to 0 if player not in penalties file
    pen_matches = pen[pen["Name"] == name]
    p_pen = float(pen_matches.iloc[0][f"{g}_penalties"]) if not pen_matches.empty else 0.0

    return {
        "name":     name,
        "pos":      row["Pos"],
        "xmins":    float(row[f"{g}_xMins"]),
        "ev":       float(row[f"{g}_Pts"]),
        "goals":    float(row[f"{g}_goals"]),  # total goals, pen contribution embedded
        "p_pen":    p_pen,                     # P(takes a penalty this GW)
        "pen_conv": pen_conv,                  # conversion rate, used only for miss risk
        "assists":  float(row[f"{g}_assists"]),
        "cs":       float(row[f"{g}_CS"]),
        "bonus":    float(row[f"{g}_bonus"]),
        "cbit":     float(row[f"{g}_cbit"]) / 100.0,
    }


def simulate(player: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Run n Monte Carlo simulations for a player in one GW.

    Scoring components:
      1. Appearance  – truncated normal around xMins → 0 / 1 / 2 pts
      2. Goals       – Poisson(λ_goals × min_scale) × position multiplier
                       λ_goals = 28_goals (total goals; Solio's penalty contribution embedded)
      3. Assists     – Poisson(λ × min_scale) × 3
      4. Clean sheet – Bernoulli(p_cs); GK/DEF need 60+ mins, MID any mins
      5. Bonus       – Poisson(λ_bonus) capped at 3 (zero-inflated naturally)
      6. CBIT        – Bernoulli(p_cbit) × 2
      7. Penalty miss – Bernoulli(p_pen × mins/90) for taken, Bernoulli(1−pen_conv) for
                        missed → −2 pts. Scored penalties already in the goals Poisson draw;
                        only the miss downside is added here since it is absent from 28_Pts.
    """
    xmins = player["xmins"]
    pos   = player["pos"]

    # Blank GW — player has no fixture
    if xmins <= 0:
        return np.zeros(n, dtype=float)

    # ── 1. Simulate actual minutes ───────────────────────────────────────────
    # Truncated normal centred at xMins (std=15) clipped to [0, 90].
    # std=15 captures rotation risk, early subs, and occasional injuries.
    raw_mins    = rng.normal(loc=xmins, scale=15.0, size=n)
    actual_mins = np.clip(raw_mins, 0.0, 90.0)

    # ── 2. Appearance points ─────────────────────────────────────────────────
    appear = np.where(
        actual_mins >= MINS_FULL, 2,
        np.where(actual_mins > 0, 1, 0),
    )

    # ── 3. Goals ─────────────────────────────────────────────────────────────
    # Use total goals lambda (Solio's projection, penalty contribution embedded).
    # Scale by actual / expected minutes to capture partial appearances.
    min_scale = np.where(actual_mins > 0, actual_mins / xmins, 0.0)
    min_scale = np.clip(min_scale, 0.0, 90.0 / xmins)
    goals     = rng.poisson(player["goals"] * min_scale)
    goal_pts  = goals * GOAL_PTS[pos]

    # ── 4. Penalty miss risk ──────────────────────────────────────────────────
    # Solio's 28_Pts does not model the −2 penalty miss deduction, so we add
    # it here.  Scored penalties are already captured in the goals Poisson draw.
    # P(miss) = P(taken) × P(not converted); only the miss branch deducts pts.
    p_pen_scaled = player["p_pen"] * (actual_mins / 90.0)
    pen_taken    = rng.binomial(1, np.clip(p_pen_scaled, 0.0, 1.0), n).astype(bool)
    pen_missed   = pen_taken & ~rng.binomial(1, player["pen_conv"], n).astype(bool)
    pen_miss_pts = pen_missed.astype(int) * PENALTY_MISS_PTS

    # ── 5. Assists ───────────────────────────────────────────────────────────
    assists    = rng.poisson(player["assists"] * min_scale)
    assist_pts = assists * ASSIST_PTS

    # ── 6. Clean sheet ───────────────────────────────────────────────────────
    played    = actual_mins > 0
    played_60 = actual_mins >= MINS_FULL
    cs_hit    = rng.binomial(1, player["cs"], n).astype(bool)
    if pos in ("G", "D"):
        cs_pts = np.where(cs_hit & played_60, CS_PTS[pos], 0)
    else:
        cs_pts = np.where(cs_hit & played, CS_PTS[pos], 0)

    # ── 7. Bonus (Poisson capped at 3) ──────────────────────────────────────
    raw_bonus = rng.poisson(player["bonus"], n)
    bonus_pts = np.minimum(raw_bonus, 3) * played

    # ── 8. CBIT bonus ────────────────────────────────────────────────────────
    cbit_pts = rng.binomial(1, player["cbit"], n) * CBIT_PTS * played

    return (
        appear + goal_pts + pen_miss_pts + assist_pts + cs_pts + bonus_pts + cbit_pts
    ).astype(float)


def print_stats(player: dict, pts: np.ndarray) -> None:
    name   = player["name"]
    ev     = player["ev"]
    thresholds = [2, 6, 10, 15]

    print(f"\n  {'─'*36}")
    print(f"  {name}  ({player['pos']})")
    print(f"  {'─'*36}")
    print(f"  Simulated mean : {pts.mean():.2f}  (file EV: {ev:.2f})")
    print(f"  Std deviation  : {pts.std():.2f}")
    print(f"  Median         : {np.median(pts):.0f}")
    for t in thresholds:
        label = f"P(≤{t} pts)" if t == 2 else f"P(≥{t} pts)"
        pct   = (pts <= t).mean() * 100 if t == 2 else (pts >= t).mean() * 100
        print(f"  {label:<16}: {pct:.1f}%")
    print(f"  90th pctile    : {np.percentile(pts, 90):.0f}")
    print(f"  95th pctile    : {np.percentile(pts, 95):.0f}")

    # Component breakdown
    pos         = player["pos"]
    p_pen       = player["p_pen"]
    pen_miss_ev = p_pen * (1 - player["pen_conv"]) * abs(PENALTY_MISS_PTS)

    print(f"\n  EV breakdown (all values in points):")
    print(f"    goals {player['goals'] * GOAL_PTS[pos]:.2f} ({player['goals']:.3f} goals×{GOAL_PTS[pos]}pt)  |  "
          f"assists {player['assists'] * ASSIST_PTS:.2f}  |  "
          f"CS {player['cs'] * CS_PTS[pos]:.2f}  |  bonus {player['bonus']:.2f}  |  "
          f"CBIT {player['cbit'] * CBIT_PTS:.2f}")
    if p_pen > 0:
        print(f"    pen miss risk: P(take)={p_pen:.2f} × P(miss)={1-player['pen_conv']:.2f}"
              f"  →  −{pen_miss_ev:.3f} exp pts  (not in Solio projection)")


def plot_comparison(
    p1: dict, pts1: np.ndarray,
    p2: dict, pts2: np.ndarray,
    gw: int,
) -> Path:
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # ── Left: overlaid distributions ────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    mmax = int(max(pts1.max(), pts2.max()))
    bins = np.arange(-0.5, min(mmax + 1.5, 32), 1)

    ax1.hist(pts1, bins=bins, density=True, alpha=0.6, color="C0", label=p1["name"])
    ax1.hist(pts2, bins=bins, density=True, alpha=0.6, color="C1", label=p2["name"])
    ax1.axvline(pts1.mean(), color="C0", linestyle="--", lw=1.5,
                label=f"{p1['name']} μ = {pts1.mean():.2f}")
    ax1.axvline(pts2.mean(), color="C1", linestyle="--", lw=1.5,
                label=f"{p2['name']} μ = {pts2.mean():.2f}")
    ax1.set_xlabel("FPL Points", fontsize=11)
    ax1.set_ylabel("Probability", fontsize=11)
    ax1.set_title(f"GW{gw} Points Distribution  (n = {len(pts1):,})")
    ax1.legend(fontsize=9)

    # ── Right: differential (colour-coded by winner) ─────────────────────────
    ax2       = fig.add_subplot(gs[0, 1])
    diff      = pts1 - pts2
    dmax      = int(max(abs(diff.min()), abs(diff.max()))) + 1
    diff_bins = np.arange(-dmax - 0.5, dmax + 1.5, 1)

    _, _, patches = ax2.hist(diff, bins=diff_bins, density=True, alpha=0.85, color="grey")

    # Colour each bar: p1-wins side → C0, p2-wins side → C1, tie bar → grey
    for patch in patches:
        x_mid = patch.get_x() + patch.get_width() / 2
        if x_mid > 0:
            patch.set_facecolor("C0")
        elif x_mid < 0:
            patch.set_facecolor("C1")

    ax2.axvline(0, color="black", lw=1.2, alpha=0.5)

    pct1  = (diff > 0).mean() * 100
    pct2  = (diff < 0).mean() * 100
    pct_t = (diff == 0).mean() * 100

    # Annotate win % directly on the relevant side
    ax2.text(0.02, 0.97, f"{p1['name']}\noutscores\n{pct1:.1f}%",
             transform=ax2.transAxes, va="top", ha="left",
             fontsize=10, fontweight="bold", color="C0")
    ax2.text(0.98, 0.97, f"{p2['name']}\noutscores\n{pct2:.1f}%",
             transform=ax2.transAxes, va="top", ha="right",
             fontsize=10, fontweight="bold", color="C1")

    ax2.set_xlabel(f"Points margin  ({p1['name']} − {p2['name']})", fontsize=11)
    ax2.set_ylabel("Probability", fontsize=11)
    ax2.set_title(f"Head-to-head margin  (tie: {pct_t:.1f}%)")

    out_path = ROOT / "tmp" / f"mc_gw{gw}_{p1['name']}_vs_{p2['name']}.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo FPL player comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gw",      type=int, required=True,   help="Gameweek number")
    parser.add_argument("--player1", type=str, required=True,   help="First player name (exact match)")
    parser.add_argument("--player2", type=str, required=True,   help="Second player name (exact match)")
    parser.add_argument("--n_sims",  type=int, default=100_000, help="Number of simulations")
    parser.add_argument("--seed",    type=int, default=42,      help="RNG seed for reproducibility")
    parser.add_argument("--no_plot",  action="store_true",        help="Suppress the matplotlib plot")
    parser.add_argument("--pen_conv", type=float, default=DEFAULT_PEN_CONV,
                        help="Penalty conversion rate (default: PL average 0.76)")
    args = parser.parse_args()

    proj = pd.read_csv(DATA_DIR / "projection_all_metrics.csv")
    pen  = pd.read_csv(DATA_DIR / "penalties.csv")

    p1 = load_player_data(proj, pen, args.player1, args.gw, args.pen_conv)
    p2 = load_player_data(proj, pen, args.player2, args.gw, args.pen_conv)

    rng  = np.random.default_rng(args.seed)
    pts1 = simulate(p1, args.n_sims, rng)
    pts2 = simulate(p2, args.n_sims, rng)

    print(f"\n{'═'*42}")
    print(f"  Monte Carlo Simulation — GW{args.gw}")
    print(f"  {args.n_sims:,} iterations  |  seed={args.seed}")
    print(f"{'═'*42}")

    print_stats(p1, pts1)
    print_stats(p2, pts2)

    diff = pts1 - pts2
    print(f"\n  {'─'*36}")
    print(f"  Head-to-head")
    print(f"  {'─'*36}")
    print(f"  {p1['name']:<20} wins: {(diff > 0).mean()*100:.1f}%")
    print(f"  {'Tie':<20}      : {(diff == 0).mean()*100:.1f}%")
    print(f"  {p2['name']:<20} wins: {(diff < 0).mean()*100:.1f}%")

    if not args.no_plot:
        out = plot_comparison(p1, pts1, p2, pts2, args.gw)
        print(f"\n  Plot saved → {out}")


if __name__ == "__main__":
    main()
