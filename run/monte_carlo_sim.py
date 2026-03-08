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
    python run/monte_carlo_sim.py --gws 28 --player1 "Gordon" --player2 "Rice"
    python run/monte_carlo_sim.py --gws 28 29 30 --player1 "Gordon" --player2 "Rice"
    python run/monte_carlo_sim.py --gws 28 29 --player1 "Haaland" --player2 "M.Salah" --pen_conv 0.78
    python run/monte_carlo_sim.py --gws 28 --player1 "Gordon" --player2 "Rice" --n_sims 200000 --no_plot
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _normalize(s: str) -> str:
    """Strip diacritics for accent-insensitive name matching (e.g. Munoz → Muñoz)."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paths import DATA_DIR  # noqa: E402

# ── FPL scoring rules ──────────────────────────────────────────────────────────
GOAL_PTS          = {"G": 6, "D": 6, "M": 5, "F": 4}
CS_PTS            = {"G": 4, "D": 4, "M": 1, "F": 0}
ASSIST_PTS        = 3
CBIT_PTS          = 2       # 2-point bonus for exceeding CBIT threshold
PENALTY_MISS_PTS  = -2      # points deducted for a missed penalty
MINS_FULL         = 60      # minutes threshold for full appearance bonus
DEFAULT_PEN_CONV  = 0.76    # Premier League average penalty conversion rate
SAVES_PER_SAVE_PT = 3       # saves threshold per +1 save point
SAVES_SLOPE       = 0.7891  # OLS: λ_saves = SAVES_SLOPE × λ_gc + SAVES_INTERCEPT
SAVES_INTERCEPT   = 1.0066  # recalibrated after separating out penalty-save EV
SAVES_FALLBACK    = 2.10    # λ_saves fallback for DGW / no fixture data
PEN_SAVE_PTS      = 5       # points for a GK penalty save
# ──────────────────────────────────────────────────────────────────────────────


def load_player_data(
    proj: pd.DataFrame,
    pen: pd.DataFrame,
    name: str,
    gw: int,
    pen_conv: float,
    fix: pd.DataFrame | None = None,
    team: str | None = None,
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

    GC model (GK/DEF, single GWs only):
      When fix (fixture_difficulty_all_metrics.csv) is provided, the team's
      projected goals conceded (λ_gc) is stored so that simulate() can draw
      GC ~ Poisson(λ_gc) and use that single draw to resolve both the clean
      sheet bonus and the −1 pt / 2 goals conceded deduction consistently.
      Double gameweeks are detected via xMins > 95 and fall back to the
      independent Bernoulli CS draw (no GC deduction), since the aggregated
      λ_gc across two fixtures cannot be applied to a single Poisson draw.
    """
    proj_matches = proj[proj["Name"] == name]
    if proj_matches.empty:
        norm = _normalize(name)
        proj_matches = proj[proj["Name"].map(_normalize) == norm]
        if proj_matches.empty:
            raise ValueError(f"Player '{name}' not found. Check spelling / capitalisation.")
        matched = proj_matches.iloc[0]["Name"]
        if matched != name:
            print(f"  [name] '{name}' matched → '{matched}'")
    if team is not None:
        team_matches = proj_matches[proj_matches["Team"] == team]
        if team_matches.empty:
            raise ValueError(f"Player '{name}' not found for team '{team}'.")
        proj_matches = team_matches
    elif len(proj_matches) > 1:
        teams = proj_matches["Team"].tolist()
        raise ValueError(
            f"Multiple players named '{name}' found (teams: {teams}). "
            "Specify team= to disambiguate."
        )
    row = proj_matches.iloc[0]
    g   = str(gw)

    # Penalty probability — default to 0 if player not in penalties file
    pen_matches = pen[pen["Name"] == name]
    p_pen = float(pen_matches.iloc[0][f"{g}_penalties"]) if not pen_matches.empty else 0.0

    xmins = float(row[f"{g}_xMins"])
    pos   = row["Pos"]

    # GC rate — for GK/DEF/MID in single GWs when fixture data is available
    # (FWD excluded: CS_PTS["F"] = 0 and no GC deduction applies)
    gc_rate: float | None = None
    if fix is not None and pos in ("G", "D", "M") and xmins <= 95:
        fix_row = fix[fix["Team"] == row["Team"]]
        if not fix_row.empty:
            rate = float(fix_row.iloc[0][f"{g}_GC"])
            if rate > 0:
                gc_rate = rate

    # Save-point rate — GK only.
    # Single GW with fixture data: λ_saves derived from λ_gc via empirical OLS fit
    #   (r=0.82, n=168 starting-GK single-GW observations).
    # DGW or no fixture data: fall back to league-average λ_saves.
    lam_saves: float = 0.0
    if pos == "G":
        if gc_rate is not None:
            lam_saves = max(0.0, SAVES_SLOPE * gc_rate + SAVES_INTERCEPT)
        else:
            lam_saves = SAVES_FALLBACK

    # Opponent penalty probability — GK only, SGW only.
    # Sum all opponent players' P(takes a penalty) for this GW.  Because penalty
    # takers within a team are mutually exclusive, this sum equals P(opponent team
    # takes at least one penalty this GW), i.e. P(GK faces a penalty).
    # Only computed for SGWs (xmins <= 95) where the opponent is unambiguous.
    p_pen_opp: float = 0.0
    if pos == "G" and fix is not None and xmins <= 95:
        fix_gk_row = fix[fix["Team"] == row["Team"]]
        if not fix_gk_row.empty:
            opp_raw  = str(fix_gk_row.iloc[0][f"{g}_OPP"])
            opp_abbr = re.sub(r"\([AH]\)", "", opp_raw).strip()
            opp_name_match = fix[fix["Abbr"] == opp_abbr]["Team"]
            if not opp_name_match.empty:
                opp_team  = opp_name_match.iloc[0]
                p_pen_opp = float(pen[pen["Team"] == opp_team][f"{g}_penalties"].sum())

    return {
        "name":      name,
        "pos":       pos,
        "xmins":     xmins,
        "ev":        float(row[f"{g}_Pts"]),
        "goals":     float(row[f"{g}_goals"]),  # total goals, pen contribution embedded
        "p_pen":     p_pen,                     # P(takes a penalty this GW)
        "pen_conv":  pen_conv,                  # conversion rate, used only for miss risk
        "assists":   float(row[f"{g}_assists"]),
        "cs":        float(row[f"{g}_CS"]),     # fallback only (DGW / no fixture data)
        "bonus":     float(row[f"{g}_bonus"]),
        "cbit":      float(row[f"{g}_cbit"]) / 100.0,
        "gc_rate":   gc_rate,                   # None → fall back to Bernoulli CS
        "lam_saves": lam_saves,                 # 0.0 for non-GKs
        "p_pen_opp": p_pen_opp,                 # P(GK faces a penalty); 0.0 for non-GKs/DGWs
    }


def simulate(player: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Run n Monte Carlo simulations for a player in one GW.

    Scoring components:
      1. Appearance  – GK: Bernoulli(xMins/90) × 90 mins (bimodal: play full game or not at all)
                       Outfield: truncated normal around xMins → 0 / 1 / 2 pts
      2. Goals       – Poisson(λ_goals × min_scale) × position multiplier
                       λ_goals = 28_goals (total goals; Solio's penalty contribution embedded)
      3. Assists     – Poisson(λ × min_scale) × 3
      4. Clean sheet / Goals conceded (GK/DEF, single GW with fixture data):
                       GC ~ Poisson(λ_gc); CS awarded when GC == 0 and played 60+ mins;
                       −floor(GC / 2) pts deducted for any playing time.
                       This unified draw ensures CS and GC deductions are self-consistent
                       (verified: exp(−λ_gc) ≈ Solio p_cs within ±0.006 for all teams/SGWs).
                     (DGW / no fixture data fallback): Bernoulli(p_cs), no GC deduction.
      5. Bonus       – Poisson(λ_bonus) capped at 3 (zero-inflated naturally)
      6. CBIT        – Bernoulli(p_cbit × min_scale) × 2; scaled by minutes since defcon%
                       is calibrated against expected mins, not a per-90 rate.
      7. Penalty miss – Bernoulli(p_pen × mins/90) for taken, Bernoulli(1−pen_conv) for
                        missed → −2 pts. Scored penalties already in the goals Poisson draw;
                        only the miss downside is added here since it is absent from 28_Pts.
      8. GK save points – Poisson(λ_saves) // 3; λ_saves derived from λ_gc via OLS
                          (λ_saves = 1.08×λ_gc + 1.01, r=0.82, 168 SGW observations).
                          Falls back to λ_saves = 2.50 for DGW / no fixture data.
                          Zero for non-GK positions.
      9. GK penalty save – Bernoulli(p_pen_opp) draws whether the opponent takes a
                           penalty; if so Bernoulli(1−pen_conv) determines whether the
                           GK saves it → +5 pts. p_pen_opp = sum of opponent players'
                           P(takes penalty), derived from the {g}_OPP fixture column and
                           penalties.csv. Zero for non-GKs and DGWs.
    """
    xmins = player["xmins"]
    pos   = player["pos"]

    # Blank GW — player has no fixture
    if xmins <= 0:
        return np.zeros(n, dtype=float)

    # ── 1. Simulate actual minutes ───────────────────────────────────────────
    if pos == "G":
        # Goalkeepers are almost never substituted — xMins ≈ P(starts) × 90.
        # Model as Bernoulli(p_play): each simulation the GK either plays the
        # full 90 minutes or doesn't feature at all.
        p_play      = np.clip(xmins / 90.0, 0.0, 1.0)
        actual_mins = np.where(rng.binomial(1, p_play, n).astype(bool), 90.0, 0.0)
    else:
        # Outfield: truncated normal centred at xMins (std=15) clipped to [0, 90].
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

    # ── 6. Clean sheet / Goals conceded ──────────────────────────────────────
    played    = actual_mins > 0
    played_60 = actual_mins >= MINS_FULL
    gc_rate   = player.get("gc_rate")

    if gc_rate is not None:
        # GK/DEF/MID single GW: unified GC draw keeps CS and GC deduction consistent.
        # exp(−λ_gc) ≈ Solio p_cs (verified within ±0.006 for all teams/SGWs).
        gc_sim = rng.poisson(gc_rate, n)
        if pos in ("G", "D"):
            # 4 CS pts, need 60+ mins; −1 pt per 2 goals conceded
            cs_pts = np.where((gc_sim == 0) & played_60, CS_PTS[pos], 0)
            gc_pts = -(gc_sim // 2) * played
        else:
            # MID: 1 CS pt, any playing minutes; no GC deduction
            cs_pts = np.where((gc_sim == 0) & played, CS_PTS[pos], 0)
            gc_pts = 0
    else:
        # Fallback: FWD, DGW, or no fixture data — independent Bernoulli CS.
        gc_sim = None
        cs_hit = rng.binomial(1, player["cs"], n).astype(bool)
        if pos in ("G", "D"):
            cs_pts = np.where(cs_hit & played_60, CS_PTS[pos], 0)
        else:
            cs_pts = np.where(cs_hit & played, CS_PTS[pos], 0)
        gc_pts = 0

    # ── 7. Bonus (Poisson capped at 3) ──────────────────────────────────────
    raw_bonus = rng.poisson(player["bonus"], n)
    bonus_pts = np.minimum(raw_bonus, 3) * played

    # ── 8. CBIT bonus ────────────────────────────────────────────────────────
    # defcon% is tied to expected mins (not a per-90 rate), so scale by
    # min_scale so that partial appearances reduce the probability proportionally.
    # min_scale is already 0 when actual_mins == 0, so no extra `played` gate needed.
    cbit_prob = np.clip(player["cbit"] * min_scale, 0.0, 1.0)
    cbit_pts  = rng.binomial(1, cbit_prob, n) * CBIT_PTS

    # ── 9. GK save points ────────────────────────────────────────────────────
    # +1 pt per 3 saves made. λ_saves is 0 for non-GKs.
    # Rate derived from λ_gc via OLS (r=0.82); fallback to league average for
    # DGWs / no fixture data. Gated by played so non-playing GKs score 0.
    saves     = rng.poisson(player["lam_saves"], n) if player["lam_saves"] > 0 else np.zeros(n, dtype=int)
    save_pts  = (saves // SAVES_PER_SAVE_PT) * played

    # ── 10. GK penalty save ───────────────────────────────────────────────────
    # p_pen_opp = P(opponent takes a penalty this GW), derived from the sum of
    # all opponent players' P(takes penalty).  If a penalty is faced, the GK
    # saves it with probability (1 − pen_conv).  Zero for non-GKs and DGWs.
    p_pen_opp = player["p_pen_opp"]
    if p_pen_opp > 0:
        pen_faced    = rng.binomial(1, np.clip(p_pen_opp, 0.0, 1.0), n).astype(bool) & played
        pen_saved_gk = pen_faced & ~rng.binomial(1, player["pen_conv"], n).astype(bool)
        pen_save_pts = pen_saved_gk.astype(int) * PEN_SAVE_PTS
    else:
        pen_save_pts = 0

    return (
        appear + goal_pts + pen_miss_pts + assist_pts + cs_pts + gc_pts + bonus_pts
        + cbit_pts + save_pts + pen_save_pts
    ).astype(float)


# ── Printing ───────────────────────────────────────────────────────────────────

def print_results(
    name1: str, pos1: str, ev_sum1: float, pts1_per_gw: list,
    name2: str, pos2: str, ev_sum2: float, pts2_per_gw: list,
    gws: list,
    n_sims: int,
    seed: int,
) -> None:
    n_gws    = len(gws)
    gw_label = f"GW{gws[0]}" if n_gws == 1 else f"GW{gws[0]}–{gws[-1]}"
    cum1     = sum(pts1_per_gw)
    cum2     = sum(pts2_per_gw)

    print(f"\n{'═'*42}")
    print(f"  Monte Carlo Simulation — {gw_label}")
    print(f"  {n_sims:,} iterations  |  seed={seed}")
    print(f"{'═'*42}")

    # Per-GW mean table (multi-GW only)
    if n_gws > 1:
        print(f"\n  Per-gameweek simulated means")
        print(f"  {'GW':<6}  {name1:<18}  {name2}")
        print(f"  {'─'*42}")
        for gw, pts1, pts2 in zip(gws, pts1_per_gw, pts2_per_gw):
            print(f"  {gw:<6}  {pts1.mean():<18.2f}  {pts2.mean():.2f}")

    # Per-player cumulative stats
    thresholds_le = [2] if n_gws == 1 else []
    thresholds_ge = [6, 10, 15] if n_gws == 1 else [10, 20, 30, 40]

    for name, pos, ev_sum, cum in [
        (name1, pos1, ev_sum1, cum1),
        (name2, pos2, ev_sum2, cum2),
    ]:
        label_suffix = "" if n_gws == 1 else "  — cumulative"
        print(f"\n  {'─'*36}")
        print(f"  {name}  ({pos}){label_suffix}")
        print(f"  {'─'*36}")
        print(f"  Simulated mean : {cum.mean():.2f}  (file EV: {ev_sum:.2f})")
        print(f"  Std deviation  : {cum.std():.2f}")
        print(f"  Median         : {np.median(cum):.0f}")
        for t in thresholds_le:
            print(f"  {'P(≤'+str(t)+' pts)':<16}: {(cum <= t).mean()*100:.1f}%")
        for t in thresholds_ge:
            print(f"  {'P(≥'+str(t)+' pts)':<16}: {(cum >= t).mean()*100:.1f}%")
        print(f"  90th pctile    : {np.percentile(cum, 90):.0f}")
        print(f"  95th pctile    : {np.percentile(cum, 95):.0f}")

    # Head-to-head summary
    diff = cum1 - cum2
    print(f"\n  {'─'*36}")
    print(f"  Head-to-head  ({gw_label})")
    print(f"  {'─'*36}")
    print(f"  {name1:<20} wins: {(diff > 0).mean()*100:.1f}%")
    print(f"  {'Tie':<20}      : {(diff == 0).mean()*100:.1f}%")
    print(f"  {name2:<20} wins: {(diff < 0).mean()*100:.1f}%")


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _plot_overlay(
    ax,
    name1: str, name2: str,
    pts1: np.ndarray, pts2: np.ndarray,
    title: str,
    show_legend: bool = True,
) -> None:
    """Overlaid points distribution for two players."""
    mmax = int(max(pts1.max(), pts2.max()))
    bins = np.arange(-0.5, min(mmax + 1.5, 120), 1)

    ax.hist(pts1, bins=bins, density=True, alpha=0.6, color="C0", label=name1)
    ax.hist(pts2, bins=bins, density=True, alpha=0.6, color="C1", label=name2)
    ax.axvline(pts1.mean(), color="C0", linestyle="--", lw=1.5,
               label=f"{name1} μ={pts1.mean():.1f}")
    ax.axvline(pts2.mean(), color="C1", linestyle="--", lw=1.5,
               label=f"{name2} μ={pts2.mean():.1f}")
    ax.set_xlabel("FPL Points", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title(title, fontsize=10)
    if show_legend:
        ax.legend(fontsize=8)


# ── Strip / range plot ────────────────────────────────────────────────────────

def _plot_strip(
    ax,
    name1: str, name2: str,
    pts1: np.ndarray, pts2: np.ndarray,
    title: str,
) -> None:
    """Horizontal strip/range plot with density-faded bars and blank/haul zones.

    For each player:
      - thin bar   : 5th–95th percentile
      - medium bar : 10th–90th percentile
      - thick bar  : 25th–75th percentile (IQR)
      - dot        : mean
      - labels     : p5, μ, p95
    Overlapping semi-transparent layers give higher colour intensity toward
    the centre, fading out to the tails.
    Background: light red zone ≤2 pts (blank potential),
                light green zone ≥8 pts (haul potential).
    """
    for pts, _, color, y in [(pts1, name1, "C0", 1), (pts2, name2, "C1", 0)]:
        p5, p10, p25, p75, p90, p95 = np.percentile(pts, [5, 10, 25, 75, 90, 95])
        mean = pts.mean()

        ax.plot([p5,  p95], [y, y], color=color, lw=2,  alpha=0.25, solid_capstyle="round")
        ax.plot([p10, p90], [y, y], color=color, lw=7,  alpha=0.20, solid_capstyle="round")
        ax.plot([p25, p75], [y, y], color=color, lw=14, alpha=0.45, solid_capstyle="round")
        ax.plot(mean, y, "o", color=color, ms=7, zorder=5)

        ax.text(mean, y + 0.22, f"μ={mean:.1f}", ha="center", va="bottom", fontsize=8, color=color)

    # Zone probabilities
    p_blank1 = (pts1 <= 2).mean() * 100
    p_blank2 = (pts2 <= 2).mean() * 100
    p_haul1  = (pts1 >= 8).mean() * 100
    p_haul2  = (pts2 >= 8).mean() * 100

    # Shaded zones drawn behind strips (zorder=0), clipped to current data x-range
    xlo, xhi = ax.get_xlim()
    if xlo < 2.5:
        ax.axvspan(xlo, 2.5, color="red", alpha=0.08, zorder=0)
        ax.text(
            (xlo + min(2.5, xhi)) / 2, 1.65,
            f"Blank ≤2\n{name1}: {p_blank1:.0f}%\n{name2}: {p_blank2:.0f}%",
            ha="center", va="top", fontsize=7.5, color="darkred",
            multialignment="center",
        )
    if xhi > 7.5:
        ax.axvspan(7.5, xhi, color="green", alpha=0.08, zorder=0)
        ax.text(
            (max(7.5, xlo) + xhi) / 2, 1.65,
            f"Haul ≥8\n{name1}: {p_haul1:.0f}%\n{name2}: {p_haul2:.0f}%",
            ha="center", va="top", fontsize=7.5, color="darkgreen",
            multialignment="center",
        )

    ax.set_yticks([0, 1])
    ax.set_yticklabels([name2, name1], fontsize=9)
    for tick, color in zip(ax.get_yticklabels(), ["C1", "C0"]):
        tick.set_color(color)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("FPL Points", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.7, 1.7)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)


# ── Stats text panel ──────────────────────────────────────────────────────────

def _plot_stats(
    ax,
    name1: str, pos1: str, ev_sum1: float, cum1: np.ndarray,
    name2: str, pos2: str, ev_sum2: float, cum2: np.ndarray,
    gws: list,
) -> None:
    """Render summary statistics as a text panel (axes hidden)."""
    ax.axis("off")
    n_gws = len(gws)
    diff  = cum1 - cum2

    thresholds_le = [2] if n_gws == 1 else []
    thresholds_ge = [6, 10, 15] if n_gws == 1 else [10, 20, 30, 40]

    def _lines(name, pos, ev_sum, cum):
        rows = [
            f"{name} ({pos})",
            "─" * 22,
            f"Mean : {cum.mean():.2f}  (EV {ev_sum:.2f})",
            f"Std  : {cum.std():.2f}",
            f"Med  : {np.median(cum):.0f}",
        ]
        for t in thresholds_le:
            rows.append(f"P(≤{t} pts) : {(cum <= t).mean()*100:.1f}%")
        for t in thresholds_ge:
            rows.append(f"P(≥{t} pts) : {(cum >= t).mean()*100:.1f}%")
        rows.append(f"5th percentile  : {np.percentile(cum, 5):.0f}")
        rows.append(f"95th percentile : {np.percentile(cum, 95):.0f}")
        return "\n".join(rows)

    ax.text(0.02, 0.97, _lines(name1, pos1, ev_sum1, cum1),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, fontfamily="monospace", color="C0")
    ax.text(0.58, 0.97, _lines(name2, pos2, ev_sum2, cum2),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, fontfamily="monospace", color="C1")

    gw_label = f"GW{gws[0]}" if n_gws == 1 else f"GW{gws[0]}–{gws[-1]}"
    pct1  = (diff > 0).mean() * 100
    pct2  = (diff < 0).mean() * 100
    pct_t = (diff == 0).mean() * 100
    h2h = (
        f"Head-to-head ({gw_label})\n"
        f"{'─' * 24}\n"
        f"{name1 + ' wins':<20} {pct1:.1f}%\n"
        f"{'Tie':<20} {pct_t:.1f}%\n"
        f"{name2 + ' wins':<20} {pct2:.1f}%"
    )
    ax.text(0.02, 0.28, h2h,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, fontfamily="monospace", color="black")

    ax.set_title("Summary Statistics", fontsize=10)


# ── Main plot dispatcher ───────────────────────────────────────────────────────

def plot_comparison(
    name1: str, pos1: str, ev_sum1: float,
    name2: str, pos2: str, ev_sum2: float,
    pts1_per_gw: list,
    pts2_per_gw: list,
    gws: list,
    n_sims: int,
) -> Path:
    """
    Single GW  → 3-panel layout   (overlay | strip | stats).
    Multi-GW   → two-row layout:
                   Row 1: per-GW overlaid distributions
                   Row 2: cumulative overlay | cumulative strip | stats
    """
    n_gws = len(gws)
    cum1  = sum(pts1_per_gw)
    cum2  = sum(pts2_per_gw)

    if n_gws == 1:
        # ── Single GW: 3-panel layout ─────────────────────────────────────────
        fig = plt.figure(figsize=(18, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                                width_ratios=[2, 2, 1.5])

        _plot_overlay(fig.add_subplot(gs[0, 0]), name1, name2,
                      pts1_per_gw[0], pts2_per_gw[0],
                      f"GW{gws[0]} Points Distribution  (n={n_sims:,})")
        _plot_strip(fig.add_subplot(gs[0, 1]), name1, name2,
                    pts1_per_gw[0], pts2_per_gw[0],
                    f"GW{gws[0]} Range")
        _plot_stats(fig.add_subplot(gs[0, 2]),
                    name1, pos1, ev_sum1, cum1,
                    name2, pos2, ev_sum2, cum2, gws)

    else:
        # ── Multi-GW: Option B two-row layout ────────────────────────────────
        fig_w    = max(14, 4.5 * n_gws)
        fig      = plt.figure(figsize=(fig_w, 10))
        gw_range = f"GW{gws[0]}–{gws[-1]}"

        # Row 1 — one overlay panel per GW
        gs_top = gridspec.GridSpec(
            1, n_gws, figure=fig,
            left=0.05, right=0.98, top=0.91, bottom=0.55,
            wspace=0.35,
        )
        for i, (gw, pts1, pts2) in enumerate(zip(gws, pts1_per_gw, pts2_per_gw)):
            _plot_overlay(
                fig.add_subplot(gs_top[0, i]),
                name1, name2, pts1, pts2,
                f"GW{gw}  (n={n_sims:,})",
                show_legend=(i == 0),
            )

        # Row 2 — cumulative overlay | strip | stats
        gs_bot = gridspec.GridSpec(
            1, 3, figure=fig,
            left=0.05, right=0.98, top=0.43, bottom=0.08,
            wspace=0.35, width_ratios=[2, 2, 1.5],
        )
        _plot_overlay(fig.add_subplot(gs_bot[0, 0]), name1, name2,
                      cum1, cum2,
                      f"Cumulative {gw_range}  (n={n_sims:,})")
        _plot_strip(fig.add_subplot(gs_bot[0, 1]), name1, name2,
                    cum1, cum2,
                    f"Cumulative Range  {gw_range}")
        _plot_stats(fig.add_subplot(gs_bot[0, 2]),
                    name1, pos1, ev_sum1, cum1,
                    name2, pos2, ev_sum2, cum2, gws)

        fig.suptitle(f"{name1}  vs  {name2}  —  {gw_range}",
                     fontsize=13, fontweight="bold", y=0.97)

    gw_tag   = f"gw{gws[0]}" if n_gws == 1 else f"gw{gws[0]}-{gws[-1]}"
    out_path = ROOT / "tmp" / f"mc_{gw_tag}_{name1}_vs_{name2}.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo FPL player comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gws",     type=int, nargs="+", required=True,
                        help="Gameweek number(s), e.g. --gws 28  or  --gws 28 29 30")
    parser.add_argument("--player1", type=str, required=True,
                        help="First player name (exact match)")
    parser.add_argument("--team1",   type=str, default=None,
                        help="Team of player1, to disambiguate duplicate names")
    parser.add_argument("--player2", type=str, required=True,
                        help="Second player name (exact match)")
    parser.add_argument("--team2",   type=str, default=None,
                        help="Team of player2, to disambiguate duplicate names")
    parser.add_argument("--n_sims",  type=int, default= 500_000,
                        help="Number of simulations")
    parser.add_argument("--seed",    type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--no_plot", action="store_true",
                        help="Suppress the matplotlib plot")
    parser.add_argument("--pen_conv", type=float, default=DEFAULT_PEN_CONV,
                        help="Penalty conversion rate (default: PL average 0.76)")
    args = parser.parse_args()

    proj = pd.read_csv(DATA_DIR / "projection_all_metrics.csv")
    pen  = pd.read_csv(DATA_DIR / "penalties.csv")
    fix  = pd.read_csv(DATA_DIR / "fixture_difficulty_all_metrics.csv")

    rng = np.random.default_rng(args.seed)

    pts1_per_gw: list[np.ndarray] = []
    pts2_per_gw: list[np.ndarray] = []
    ev_sum1 = ev_sum2 = 0.0
    pos1 = pos2 = ""

    for gw in args.gws:
        p1 = load_player_data(proj, pen, args.player1, gw, args.pen_conv, fix, team=args.team1)
        p2 = load_player_data(proj, pen, args.player2, gw, args.pen_conv, fix, team=args.team2)
        pts1_per_gw.append(simulate(p1, args.n_sims, rng))
        pts2_per_gw.append(simulate(p2, args.n_sims, rng))
        ev_sum1 += p1["ev"]
        ev_sum2 += p2["ev"]
        pos1 = p1["pos"]
        pos2 = p2["pos"]

    print_results(
        args.player1, pos1, ev_sum1, pts1_per_gw,
        args.player2, pos2, ev_sum2, pts2_per_gw,
        args.gws, args.n_sims, args.seed,
    )

    if not args.no_plot:
        out = plot_comparison(
            args.player1, pos1, ev_sum1,
            args.player2, pos2, ev_sum2,
            pts1_per_gw, pts2_per_gw,
            args.gws, args.n_sims,
        )
        print(f"\n  Plot saved → {out}")


if __name__ == "__main__":
    main()
