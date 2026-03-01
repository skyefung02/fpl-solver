# MID Calibration Pipeline — Experiment Documentation

> **Purpose of this document:** Fully specify the calibration experiment pipeline
> so that any script can be reconstructed from scratch without retaining the source files.
> Includes architecture, logic, parameters, formulas, and all key findings.

---

## 1. Context and Motivation

The Monte Carlo simulator (`run/monte_carlo_sim.py`) was found to systematically
**overestimate** MID expected points (EV) relative to Solio's projection file (`{gw}_Pts`).
The gap was largest for high-CBIT (ball-interception) defensive midfielders, suggesting a
miscalibration in the CBIT scoring component.

**Confirmed gap (sim_mean − file_ev), GW29–38, SGW only, xMins 60–95:**

| Position | Mean gap/GW |
|----------|------------|
| GK       | −0.015     |
| DEF      | +0.032     |
| FWD      | +0.054     |
| **MID**  | **+0.147** |

---

## 2. Pipeline Architecture

The five scripts form a sequential pipeline:

```
yc_gap_experiment.py
    (Monte Carlo runs → discovers +0.147 MID gap per GW)
          ↓
ev_reconstruction_expanded.py
    (Analytical, cohort-level → confirms CBIT is the primary driver)
          ↓
ev_recon_all_mids.py
    (Analytical, full MID population → writes tmp/ev_recon_all_mids.csv)
          ↓         ↙
cbit_calibration.py          plot_mid_calibration.py
    (Fits correction formula)    (Scatter plot visualisation)
```

---

## 3. Script Specifications

### 3.1 `yc_gap_experiment.py` — Gap Discovery via Monte Carlo

**Purpose:** Establish that a systematic gap exists between `sim_mean` and `file_ev` for DEF
and MID, and estimate whether yellow cards could explain it.

**Inputs:**
- `data/projection_all_metrics.csv`
- `data/penalties.csv`
- `data/fixture_difficulty_all_metrics.csv`

**Output:** `tmp/yc_gap_experiment.csv` — one row per player-GW.

**Key parameters:**
- `N_SIMS = 100_000`
- `SEED = 42`
- `GWS = range(29, 39)`
- `POSITIONS = ("D", "M")`
- `MIN_XMINS = 60.0` — only confident starters
- `DGW_THRESH = 95.0` — xMins above this → DGW, excluded

**Core logic:**
1. For each player-GW where `60 < xMins <= 95`, call `load_player_data()` then `simulate()`.
2. Compute `gap = sim_mean − file_ev`.
3. Compute `implied_p_yc_per90 = gap / (xmins / 90)` — the per-90 yellow-card rate that
   would fully explain the gap if Solio prices YC risk and the simulator does not.

**Output columns:** `Name, Pos, Team, GW, xMins, file_ev, sim_mean, gap, implied_p_yc_p90`

**Report structure:**
- Per-position gap summary (mean, median, std, % overestimated, implied mean p_yc/90)
- Per-player aggregate table (top 20 and bottom 10 by mean gap)

---

### 3.2 `ev_reconstruction_expanded.py` — Cohort Deep-Dive (Analytical)

**Purpose:** Confirm that CBIT% is the structural driver of overestimation, using two
hand-picked cohorts and a full per-component analytical breakdown.

**Inputs:**
- `data/projection_all_metrics.csv`
- `data/penalties.csv`
- `data/fixture_difficulty_all_metrics.csv`

**Output:** Console report only (no CSV).

**Key parameters:**
- `STD = 15.0` — σ for the clipped-Normal minutes distribution
- `GWS = range(29, 39)`
- Exclude xMins <= 0 or xMins > 95 (DGWs)

**Two cohorts:**

*Cohort A — Elite attacking MIDs (expected: reconstructed < file_ev):*
Salah (Liverpool), Palmer (Chelsea), Saka (Arsenal), Gordon (Newcastle),
B.Fernandes (Man Utd), Mbeumo (Man Utd), Gakpo (Liverpool), Sarr (Crystal Palace)

*Cohort B — High-CBIT defensive MIDs (expected: reconstructed > file_ev):*
J.Palhinha (Spurs), Sangaré (Nott'm Forest), André (Wolves), Casemiro (Man Utd),
Caicedo (Chelsea), Anderson (Nott'm Forest), Rice (Arsenal), M.Fernandes (West Ham),
J.Gomes (Wolves), Janelt (Brentford)

**Analytical model — exact clipped-Normal statistics:**

For `X = clip(Normal(μ, σ), 0, 90)`:
```
a = (0 − μ) / σ
b = (90 − μ) / σ
E[X]      = μ(Φ(b)−Φ(a)) − σ(φ(b)−φ(a)) + 90·(1−Φ(b)) + 0·Φ(a)
p_played  = 1 − Φ(a)          (P(raw > 0))
p_played60 = 1 − Φ((60−μ)/σ)
p_part    = Φ((60−μ)/σ) − Φ(a)   (P(0 < raw < 60))
e_ms      = E[X] / μ              (expected min_scale)
```

**EV component formulas (matching simulator exactly):**

| Component | Formula |
|-----------|---------|
| Appearance | `2 × p_played60 + 1 × p_part` |
| Goals | `GOAL_PTS[pos] × goals_rate × e_ms` |
| Assists | `ASSIST_PTS × assist_rate × e_ms` |
| CS (with gc_rate) | `CS_PTS[pos] × exp(−gc_rate) × p_played` |
| GC deduction (GK/DEF only) | `−E[floor(Poisson(gc_rate)/2)] × p_played` via series sum k=1..24 |
| Bonus | `E[min(Poisson(λ_bonus), 3)] × p_played` via layer-cake decomp |
| CBIT | `CBIT_PTS × cbit × e_ms` |
| Penalty miss | `PENALTY_MISS_PTS × p_pen × (e_mins/90) × (1 − pen_conv)` |

*E[min(Poisson(λ),3)] layer-cake:* `3 − e^{−λ}(3 + 2λ + λ²/2)`

**Report per cohort:**
- Summary table: Name, Team, n_GWs, recon_sum, file_sum, gap, gap/GW, CBIT%_of_recon, mean_cbit%
- GW-by-GW component breakdown for the first player in each cohort

---

### 3.3 `ev_recon_all_mids.py` — Full-Population Analytical Reconstruction

**Purpose:** Run the same analytical reconstruction across every MID in the projection file,
to verify the CBIT→overestimation pattern holds population-wide (not just for the hand-picked
cohort). Produces the CSV that feeds the two downstream scripts.

**Inputs:**
- `data/projection_all_metrics.csv`
- `data/penalties.csv`
- `data/fixture_difficulty_all_metrics.csv`

**Output:** `tmp/ev_recon_all_mids.csv`

**Key parameters:**
- `STD = 15.0`
- `GWS = range(29, 39)`
- `MIN_XMINS = 60.0`, `DGW_MAX = 95.0`

**Output CSV columns:**
`Name, Team, GW, xMins, cbit_pct, recon, file_ev, gap, cbit_ev`

Where `cbit_ev = CBIT_PTS × (cbit_pct/100) × e_ms`.

**Report structure:**
1. Population-level stats: n players, n obs, mean gap, % overestimated, Pearson r(mean_cbit, mean_gap)
2. Full per-player table sorted by mean gap (descending): n_obs, mean_gap, mean_cbit%, cbit_share%, mean_file_ev
3. CBIT quartile breakdown: for each of Q1–Q4, show cbit% range, n, mean_gap, median_gap

**Note on cs_ev for MIDs:** Uses `CS_PTS["M"] × exp(−gc_rate) × p_played` (no p_played60).
No GC deduction for MIDs.

---

### 3.4 `cbit_calibration.py` — Correction Formula Fitting

**Purpose:** Given the full-population reconstruction, reverse-engineer the "true" CBIT
probability implied by each observation, then fit and validate candidate correction formulas.

**Inputs:** `tmp/ev_recon_all_mids.csv` (must run `ev_recon_all_mids.py` first)

**Output:** Console report only.

**Key constants:**
- `STD = 15.0`
- `CBIT_PTS = 2`
- `BASELINE = −0.036` — mean gap for zero-CBIT players (non-CBIT baseline offset)

**Step 1 — Implied true CBIT probability:**
```
gap = (cbit_ev_current − cbit_ev_true) + BASELINE
=> cbit_ev_true  = cbit_ev_current − (gap − BASELINE)
=> implied_prob  = (raw_prob − (gap − BASELINE) / (CBIT_PTS × e_ms)).clip(lower=0)
```
Where `e_ms = E[min_scale]` is computed analytically from xMins using the same
clipped-Normal formula as above.

Per-player means computed for players with n_obs ≥ 3 (91 players, 787 obs in the dataset).

Report: Pearson r(raw_prob, implied_prob), table of raw% vs implied% vs ratio per player
(sorted by mean cbit% descending).

**Step 2 — Candidate formula benchmark:**

Formulas evaluated (raw_cbit → calibrated_cbit):
- `current`: identity (c)
- `× 0.50`: linear scale
- `× 0.33`: linear scale
- `^1.5`: power law (alias `c^1.5 = sqrt(c) * c`)
- `^2.0`: square
- `cap 15%`: `min(c, 0.15)`
- `cap 10%`: `min(c, 0.10)`
- `MM k=3/5/8/11`: Michaelis-Menten `c / (1 + k×c)`

For each formula, recompute `new_cbit_ev = CBIT_PTS × formula(raw_prob) × e_ms`, then
`new_gap = gap − cbit_ev + new_cbit_ev`. Report: mean_gap, % overestimated, RMSE,
Q1/Q2/Q3/Q4 gap breakdown by CBIT% quartile.

**Step 3 — Fitted Michaelis-Menten curve:**
Free-parameter fit: `implied = raw / (1 + k×raw)`, optimised via scipy `curve_fit`,
bounds `(0, 50)`, p0=5.0. Reports fitted k, saturation ceiling, resulting mean gap.

**Step 4 — Fitted power law:**
Free-parameter fit: `implied = raw^exp`, bounds `(1.0, 10.0)`, p0=2.0.
Reports fitted exponent, resulting mean gap.

**Step 5 — Before/after table for key high-CBIT DMs:**
Uses the MM-fitted formula. Shows old_cbit_ev, new_cbit_ev, delta, old_gap, new_gap for:
Palhinha, Sangaré, André, Casemiro, Caicedo, Anderson, Rice, M.Fernandes, J.Gomes, Janelt.

---

### 3.5 `plot_mid_calibration.py` — Scatter Plot Visualisation

**Purpose:** Visual sanity check — scatter of `mean_recon` vs `mean_file_ev` per player,
coloured by CBIT%, to confirm the overestimation pattern is systematic.

**Inputs:** `tmp/ev_recon_all_mids.csv`

**Output:** `tmp/mid_calibration_scatter.png` (dpi=160)

**Aggregation:** Per-player mean over eligible SGW observations: `n_obs, mean_file, mean_recon, mean_gap, mean_cbit`.

**Plot design:**
- Axes: x = `mean_file_ev`, y = `mean_recon` (analytical)
- Colour map: `RdYlBu_r` (blue=low CBIT, red=high CBIT), normalised 0 → max CBIT%
- Reference line: `y = x` (dashed black, α=0.6) — perfect calibration
- Shaded regions: salmon above y=x (overestimation), lightblue below (underestimation)
- Colour bar: "Mean CBIT%"
- Point size: s=55, white edge, linewidth=0.5
- Text annotations for labelled players (fontsize=6.5)

**Labelled players:**
- High-CBIT DMs: Anderson, Caicedo, Garner, J.Gomes, Casemiro, M.Fernandes,
  J.Palhinha, Sangaré, André, Rice, Ampadu, Rodrigo, Janelt, Adams, Xhaka
- Low-CBIT attackers: M.Salah, Palmer, Saka, Gordon, B.Fernandes, Mbeumo, Gakpo
- Outliers: Armstrong, Diarra, Laurent, Gruev

**Inset text box (monospace, top-left, α=0.8):**
n, total obs, mean gap, % overestimated, `r(CBIT%, gap) = 0.397`

Also prints a per-player table (sorted by mean_gap desc) to console.

---

### 3.6 `component_breakdown_sim.py` — Single-Player Component Debugger

**Purpose:** Deep-dive diagnostic for a single hardcoded player-GW. Confirms the simulator
is internally consistent (sim ≈ analytical) and identifies which component drives any gap
vs. Solio's file EV. Also tests sensitivity to the minutes σ assumption.

**Inputs:**
- `data/projection_all_metrics.csv`
- `data/penalties.csv`
- `data/fixture_difficulty_all_metrics.csv`

**Output:** Console report only.

**Key parameters:**
- `PLAYER = "Anderson"`, `TEAM = "Nott'm Forest"`, `GW = 29` (hardcoded; change as needed)
- `N_SIMS = 500_000`, `SEED = 42`
- `STD_VALUES = [10, 15, 20, 25]` — minutes σ values for sensitivity sweep

**Three sections in the report:**

*1. Component table (std=15):*
For each component (appearance, goals, assists, CS, GC deduction, bonus, CBIT, penalty miss),
print the raw Solio input (e.g. `λ_bonus=0.32`, `p_cbit=32.1%`), the analytical EV, the
simulated mean, `sim − file_ev`, and `sim − analytical`. Confirms internal consistency.

*2. Minutes distribution diagnostics:*
Prints `E[actual_mins]`, `P(≥60)`, `P(>0)`, `P(=90 cap)`, `E[min_scale]`, `std(actual_mins)`,
each paired with the analytical counterpart.

*3. Std sensitivity (both analytical and MC):*
For each σ ∈ {10, 15, 20, 25}: e_mins, e_ms, P(≥60), appear EV, goals EV, CBIT EV, bonus EV,
total EV, gap vs file_ev. Tests whether σ=15 is causing systematic bias.

**Simulate_components logic (mirrors `monte_carlo_sim.simulate` exactly):**
- `raw_mins = Normal(xmins, std)` → clipped to [0, 90]
- `min_scale = clip(actual_mins / xmins, 0, 90/xmins)`
- Goals/assists: `Poisson(λ × min_scale)` per sim
- CS/GC: if `gc_rate` available → `Poisson(gc_rate)` → CS if `gc_sim==0`, GC deduction `−floor(gc_sim/2)`; else Bernoulli CS
- Bonus: `min(Poisson(λ_bonus), 3) × played`
- CBIT: `Binomial(1, clip(cbit × min_scale, 0, 1)) × 2`
- Penalty miss: `Binomial(1, clip(p_pen × mins/90, 0, 1))` then miss check `× −2`

**Role in investigation:** Early diagnostic used to confirm the simulator is correct before
concluding the gap is a formula problem (CBIT), not simulation noise. Anderson GW29 was chosen
as a high-CBIT DM representative.

---

## 4. Key Findings

### 4.1 Root Cause of MID Overestimation

The simulator uses `cbit_prob = player["cbit"] × min_scale` where `player["cbit"]` is the
raw Solio cbit% divided by 100 (e.g. 0.322 for Palhinha at 32.2%). Treating this raw
fraction directly as a Bernoulli probability massively overstates EV for high-CBIT DMs.

**Evidence:**
- r(raw_cbit%, mean_gap) at player level: **0.397** (scatter plot annotation)
- r(raw_prob, implied_true_prob) from cbit_calibration: **0.953** — CBIT is the primary driver

### 4.2 Correction Formula Results

Benchmark (GW29-38, 787 obs, SGW, xMins 60-95):

| Formula | Mean gap/GW | RMSE | Q1 gap | Q2 gap | Q3 gap | Q4 gap |
|---------|------------|------|--------|--------|--------|--------|
| current (c) | +0.147 | 0.190 | — | — | — | — |
| ^1.5 | +0.006 | 0.114 | +0.049 | +0.049 | −0.020 | −0.050 |
| MM k≈5 (fitted) | ~+0.006 | similar | — | — | — | — |

**OLS-fitted power law exponent: 1.47** (round to 1.5 for implementation)
**Best formula: `cbit_prob = raw_cbit^1.5 × min_scale`**

### 4.3 Remaining Baseline Gap

After fixing CBIT, a residual ~+0.05/GW offset remains even for zero-CBIT players.
Hypothesis: Solio prices in yellow-card risk (−1 pt), but the simulator does not.
The implied p_yc/90 needed to explain this: ~0.05–0.06/game, plausible for a league average.

### 4.4 Elite Attacker "Underestimation" Was a DGW Artefact

Earlier analysis appeared to show Salah etc. being underestimated. This was caused by
GW33 (DGW for Salah) being included: `xMins=162.5` but the model caps at 90. Filtering
`xMins > 95` removes all DGWs and eliminates this artefact.

### 4.5 DEF Calibration is Principled

The GC deduction for DEF uses `E[floor(Poisson(gc_rate)/2)]` where `gc_rate` comes from
Solio's own `{gw}_GC` projections. This is not reverse-engineered; it mirrors the exact
FPL scoring rule applied to Solio's own expected goals-conceded, giving principled DEF calibration.

### 4.6 Status at Time of Archiving

The `^1.5` power-law fix has **not yet been implemented** in `monte_carlo_sim.py` (pending
visual inspection and further experiments). The quartile gap after the fix (+0.049 for low-CBIT
players) suggests the remaining offset is attributable to yellow cards rather than CBIT.

---

## 5. Data Dependencies

| File | Role |
|------|------|
| `data/projection_all_metrics.csv` | Solio projections: Name, Team, Pos, `{gw}_xMins`, `{gw}_Pts`, `{gw}_goals`, `{gw}_assists`, `{gw}_CS`, `{gw}_bonus`, `{gw}_cbit`, `{gw}_GC` |
| `data/penalties.csv` | Per-player per-GW penalty probability: `{gw}_penalties` |
| `data/fixture_difficulty_all_metrics.csv` | `{gw}_GC`, `{gw}_OPP`, `Abbr`, `Team` — used to get `gc_rate` per team per GW |
| `tmp/ev_recon_all_mids.csv` | Intermediate: written by `ev_recon_all_mids.py`, read by `cbit_calibration.py` and `plot_mid_calibration.py` |
| `tmp/mid_calibration_scatter.png` | Plot output |
| `tmp/yc_gap_experiment.csv` | MC gap results |

---

## 6. Simulator Constants (at time of experiments)

These were imported from `run/monte_carlo_sim.py`:

| Constant | Value |
|----------|-------|
| `GOAL_PTS` | `{"G":6,"D":6,"M":5,"F":4}` |
| `CS_PTS` | `{"G":6,"D":6,"M":1,"F":0}` |
| `ASSIST_PTS` | 3 |
| `CBIT_PTS` | 2 |
| `DEFAULT_PEN_CONV` | (from simulator — typical ~0.75) |
| `PENALTY_MISS_PTS` | −2 |
| Minutes σ | 15.0 |
