# EO Rank Bands — Design Plan

> **Purpose of this document:** Specify the extension of `run/effective_ownership.py` from a
> single top-10k cohort to a set of disjoint rank strata, so that transfer-flow features can be
> regressed against EO *as a function of rank* rather than at one point on that curve.
>
> Status: **band plumbing implemented 2026-08-27** (steps 1 of section 8). Steps 2-5 outstanding.
> Written 2026-08-27.

---

## 1. Motivation

The training target is next-gameweek EO for the top-10k cohort. The primary feature is transfer
flow (`transfers_in_event` / `transfers_out_event` from `bootstrap-static`), which is **global** —
aggregated across all ~11m managers.

With only the top-10k band, the model must learn the global→elite mapping from the time
dimension alone: ~37 gameweek transitions per season. That is a hard extrapolation from one
scalar to another, estimated from very few independent observations.

Sampling multiple strata changes the identification strategy:

1. **Cross-sectional identification of the rank response.** The same transfer wave is observed
   landing at every rank level within a single gameweek. The rank-dependence is then estimated
   from cross-section, not soaked up by the ~37 time transitions.

2. **A shape feature that is otherwise unconstructible.** For each player, EO across strata forms
   a curve. Its slope — the *elite premium* — distinguishes an elite-led move from a mass
   bandwagon. "João Pedro bought by 170k" means something entirely different when elite ownership
   is climbing ahead of the mass versus when the elite band is already selling into it. This
   cannot be built from a single band at any sample size.

3. **~5× training rows**, though not independent: cluster cross-validation folds by gameweek.

---

## 2. Time-critical dependency

**The GW1 band backfill must happen before GW2 matches kick off.**

Cohort membership cannot be reconstructed retroactively. The standings endpoint serves only the
*current* ranking, so "who was in the top 500k at GW1" is unrecoverable once GW2 results land.
Measuring today's top-500k against GW1 picks would reintroduce exactly the conditioning-on-results
error that `--from-cohort` exists to avoid (see the module docstring in
`run/effective_ownership.py`).

Until GW2 matches are played, the live Overall standings **are** the post-GW1 standings, so a pull
now reproduces exactly what the GW1 t=0 band pull would have been.

| Fact | Value |
|---|---|
| GW1 | finished, `data_checked` true |
| GW2 deadline | 2026-08-28T17:30:00Z |
| Clock at time of writing | 2026-08-27T22:32Z |
| Window remaining | ~19 h |

**Knock-on effect:** `run_auto()` only issues a label pull for `gw > 1`, and that pull loads
`cohort_gw1_<band>.json`. Without the GW1 band cohorts frozen before kickoff, the first per-band
labels cannot exist until **GW3** — a whole gameweek of training data lost on top of GW1.

The existing `eo_gw1_top10000.csv` / `cohort_gw1_top10000.json` already cover the 1–10k stratum.
Only the four new strata need backfilling.

---

## 3. Design decisions

### 3.1 Disjoint strata, not nested pools

`--pool 100000` today means ranks 1–100,000, which *contains* the top 10k. Cumulative bands are
recoverable from strata by weighted averaging; strata are **not** cleanly recoverable from
cumulative bands, because subtracting two sampled estimates amplifies their noise. Sample the
strata directly.

| Stratum | Positional range | Pages | Sampled pages | Managers | Priority |
|---|---|---|---|---|---|
| `top10000`      | 1 – 10,000        | 200    | 200 (full) | 10,000 | target band |
| `r10k-30k`      | 10,001 – 30,000   | 400    | 100        | 5,000  | high |
| `r30k-100k`     | 30,001 – 100,000  | 1,400  | 120        | 6,000  | high |
| `r100k-250k`    | 100,001 – 250,000 | 3,000  | 120        | 6,000  | high |
| `r250k-500k`    | 250,001 – 500,000 | 5,000  | 100        | 5,000  | medium |
| `r500k-1m`      | 500,001 – 1,000,000 | 10,000 | 100      | 5,000  | medium |

`r10k-30k` and `r30k-100k` replace an original single `r10k-100k` stratum, split once the GW1
design effects came in (§6). That band spanned a 10× rank range — by far the widest in log terms —
across the steepest part of the curve, where De Cuyper runs 37.5 → 16.2 → 7.8 at its boundaries.
Measured clustering turned out to be near-free (DEFF ≈ 1.1), which makes finer strata the
efficient way to spend request budget, so the split costs ~2,100 requests per pull and resolves
the steep region into two points instead of one.

`r10k-100k` remains defined in `BANDS` — the registry is deliberately not a partition — so its
GW1 pull stays addressable and reproducible. It is not in `DEFAULT_BANDS`, which is disjoint and
contiguous. It also serves as a consistency check: the rank-weighted average of the two sub-bands
should reproduce it.

Cumulative bands remain derivable post-hoc as a size-weighted mean over strata.

### 3.2 The 500k–1m stratum

In the default set. It was originally scoped as opt-in on the argument that
`selected_by_percent` — already pulled from `bootstrap-static` every run and stored as
`global_owned` — anchors the far end of the ownership curve for free, and that the curve is
nearly flat past 500k.

That argument does not survive contact with what the band is for. `selected_by_percent` is
**ownership only**: no captaincy, no bench multipliers. GW1 showed captaincy carrying signal that
ownership does not — Haaland's captaincy climbs 21.8% → 35.2% across the bands while his
ownership moves far less — so the free proxy cannot stand in for a measured band at the far end.
It is also a **live snapshot** rather than a deadline-frozen measurement, which makes it
non-comparable with every other band in the set.

Cost is ~10,100 requests per gameweek (~2 min), against a full-set runtime that stays well inside
the timeout. The band is cheap, and it is the only anchor for the mass end of the captaincy curve.

### 3.3 Stratified systematic page sampling

`select_pages()` currently uses `random.Random(seed).sample(...)`. Drawing 100 pages uniformly at
random from 10,000 leaves large rank gaps by chance. Replace with a systematic draw: step
`k = total_pages / n_sample`, take pages at `lo_page + floor(i*k + offset)`, with `offset`
derived from the seed. Even rank coverage, identical cost.

### 3.4 Fixed page positions across gameweeks

Derive the sampling offset from the **band identity only**, not from the gameweek. Holding the
sampled rank slices fixed all season makes the design paired: week-over-week *changes* in EO —
which is what the model predicts — have much of the cluster noise cancel out. Re-randomising each
week would add independent cluster noise to every delta.

Note this fixes the *rank slices*, not the managers. Following the same managers is the separate
axis already handled by `--from-cohort`.

### 3.5 Band by position, not by `rank`

Early-season ties are enormous: page 200 currently returns 50 entries all reporting
`rank: 8722`, and page 20,000 reports `rank: 893714` against `rank_sort: 999951`. Define strata by
**positional index** (`page * 50`), which is what pagination actually indexes. Record `rank_sort`
for reference.

---

## 4. Code changes

All in `run/effective_ownership.py` unless noted.

### 4.1 Band representation

Introduce a band descriptor — a small frozen dataclass or named tuple:

```python
Band(name: str, lo: int, hi: int, sample_pages: int | None)
```

`lo`/`hi` are 1-based inclusive positional ranks. `sample_pages=None` means enumerate fully.
Define the five bands as a module-level `BANDS` tuple.

### 4.2 `select_pages()`

Signature becomes `select_pages(band: Band, seed: int) -> list[int]`. Page range is
`ceil(lo/50) .. ceil(hi/50)`. Systematic draw per §3.3; offset seeded on `(seed, band.name)` per
§3.4 so it is stable across gameweeks.

### 4.3 `tag_for()`

Takes a `Band` instead of `(pool, sample_pages)`. **Must emit the legacy stem `gw{gw}_top10000`
for the 1–10k band**, otherwise the `out_path.exists()` guard in `run_pull()` stops matching and
GW1 is needlessly re-pulled. New strata use `gw{gw}_{band.name}_sampled`, with the existing
`_cohort{N}` suffix appended for label pulls.

### 4.4 `load_cohort()`

Keyed on band so a label pull reloads the cohort frozen for that same stratum.

### 4.5 `resolve_scope()` → `resolve_bands()`

Replaced by `resolve_bands()`. `--pool`, `--100k` and `--sample-pages` were retired outright —
nothing else in the repo imports this module and the tests do not touch it — and replaced by
`--bands` (comma-separated names, default: the four high-priority strata).

### 4.6 `run_pull()`

Takes a `Band`. Add a `band` column to the output frame in `build_table()` so per-band CSVs concat
cleanly. Add `band`, `rank_lo`, `rank_hi` to the metadata JSON.

### 4.7 `run_auto()`

Inner loop over bands: for each gameweek in the window, for each band, attempt the label pull
(`gw > 1`) then the t=0 pull. Gates are unchanged and per-band skips stay independent.

### 4.8 Timeout

Steady state fits comfortably; the catch-up case does not (§5). Default raised from 45 to 150
minutes, which covers the full 3-gameweek catch-up with headroom. One process per band remains
the fallback if the wide strata later need throttling.

### 4.9 Margin-of-error reporting

The printed MoE assumes an independent sample. Sampled bands are clusters of 50 adjacent ranks, so
it should divide by the design effect (§6).

### 4.10 `launchd/com.skyefung.fpl-eo.plist`

No structural change if the timeout is raised. If bands run as separate processes, either add
`--bands` to `ProgramArguments` per job or wrap in a small shell driver.

---

## 5. Cost

Throughput assumption: 2.3 req/s/worker × 24 workers = 55.2 req/s (the script's own estimator).

| Stratum | t=0 requests | label requests |
|---|---|---|
| `top10000`   | 10,200 | 10,000 |
| `r10k-30k`   |  5,100 |  5,000 |
| `r30k-100k`  |  6,120 |  6,000 |
| `r100k-250k` |  6,120 |  6,000 |
| `r250k-500k` |  5,100 |  5,000 |
| `r500k-1m`   |  5,100 |  5,000 |
| **Total (default set)** | **37,740** | **37,000** |

Per gameweek: **74,740 requests**. At the 55.2 req/s the estimator assumes, that is 11.4 min for
the t=0 pull, 11.2 min for the label pull, and 68 min for a 3-gameweek catch-up — inside the
150 min timeout. At the ~88 req/s actually observed, 14.2 min per gameweek and 42 min for a
catch-up.

Deep pagination was verified working: page 20,000 returns HTTP 200 with `has_next: true`.

---

## 6. Precision and validation

Sampled bands are **cluster samples** — each page is 50 adjacent ranks — so the
independent-sample margin of error overstates precision. Two design effects are now measured on
every sampled pull by `estimate_clustering()`, at no extra request cost:

- `deff_cluster` — the textbook exchangeable-cluster ratio, and the ICC that
  `DEFF = 1 + (m−1)·ICC` is defined against.
- `deff_systematic` — the successive-difference estimator, which uses variance between
  *adjacent* sampled pages. This is the one that applies to our design: page order is rank order
  and `select_pages()` spreads the sample evenly, so a smooth ownership-vs-rank gradient is
  sampled like a stratified design rather than paying the full cluster penalty.

Validated on synthetic data before use — independent data returns DEFF 0.99 from both estimators;
a smooth gradient returns 5.39 from the cluster form but 0.88 from the systematic form; a random
per-page shock returns ~3.5 from both. It correctly separates a gradient that systematic sampling
handles from real clustering that it does not.

**Measured at GW1:**

| Band | n | `deff_cluster` | `icc_cluster` | `deff_systematic` | n_eff | MoE at p=0.5 |
|---|---|---|---|---|---|---|
| `r10k-100k`  | 9,000 | 1.314 | 0.0064 | 1.085 | 8,295 | ±1.08 pp |
| `r100k-250k` | 6,000 | 1.399 | 0.0082 | 1.189 | 5,046 | ±1.38 pp |
| `r250k-500k` | 5,000 | 1.399 | 0.0082 | 1.151 | 4,344 | ±1.49 pp |
| `r500k-1m`   | 5,000 | 1.252 | 0.0051 | 1.119 | 4,468 | ±1.47 pp |

The originally assumed ICC of 0.02 (DEFF 1.98) was roughly twice too pessimistic. `ASSUMED_ICC`
is now 0.007 and serves only as a fallback when page labels are unavailable.

**Consequence for the page budgets.** They need no increase — at DEFF ≈ 1.1 the bands are
already far more precise than the effects being measured (band-to-band ownership gaps run 5–30 pp
against a ±1.5 pp margin). Note also that the earlier "more pages × fewer managers per page"
trade is not available: a standings page is an atomic 50 entries per request, so total pages is
the only lever. Since clustering is nearly free, each additional page buys ~46 effective
managers, which makes *finer strata* — rather than bigger ones — the efficient way to spend
any further request budget.

This matters because these bands feed the model as features and, for the top-10k band, as the
label. Noise in features attenuates coefficients; noise in the label inflates variance. Keeping
`top10000` fully enumerated (no sampling) is the correct allocation of the request budget and is
already the current behaviour.

---

## 6a. Chip-rate confound in cross-band comparison (observed at GW1)

A t=0 cohort is ranked by the gameweek it measures, so it is selection-enriched for whatever
inflated that gameweek's score. At GW1 that is bench boost, and the enrichment is stronger the
higher the band:

| Band | bboost | 3xc | none |
|---|---|---|---|
| `top10000`  | 73.6% | 10.3% | 16.2% |
| `r10k-100k` | 59.6% |  5.7% | 34.7% |

Because bench boost lifts every bench pick to multiplier ≥ 1, a band with more bench boosts has
systematically higher EO. Measured at GW1, the effect is a near-uniform additive offset rather
than a per-player distortion:

- `corr(eo_premium, ownership_premium) = 0.964`
- mean EO premium `+0.128` vs mean ownership premium `+0.001`

**Implication for the shape feature (§1.2).** Build the elite premium on `owned` and `captain`
rather than raw `eo`. EO premium decomposes as ownership premium + captaincy premium + chip
offset; the first two are the signal, the third is an artefact of cohort selection. Where the two
diverge per player it is captaincy, not chips — at GW1, João Pedro shows a `+4.36` EO premium on
only `+0.59` ownership premium, which is genuine elite captaincy concentration.

`owned` is chip-independent and sums to exactly 1500 (15 picks × 100%) in every band, which also
makes it a useful integrity check on a pull.

---

## 7. Risks

- **API load.** ~35k requests twice weekly, up from ~10k. `build_session()` already retries on 429
  with backoff. Consider lowering `--workers` for the wide strata if 429s appear in
  `~/Library/Logs/fpl-eo.log`.
- **Pagination depth could be capped by FPL later.** `fetch_cohort()` already counts and reports
  failed standings pages; that degradation path should stay intact so a capped band yields a
  short cohort rather than an aborted run.
- **Band composition drifts within a season.** A fixed rank slice contains different managers each
  week by construction. That is intended for EO(rank); the manager-following axis is
  `--from-cohort`.

---

## 8. Execution order

1. ~~**Before GW2 kickoff (~19 h):** implement bands far enough to run t=0 pulls, and freeze GW1
   cohorts for the four new strata. This is the only irreversible item.~~ **Done 2026-08-27.**
2. Measure ICC per band from that GW1 data; re-tune page counts (§6). `pages` is now recorded in
   the pull metadata alongside `entry_ids` (stored in page order, 50 per page), so intra-page
   variance is computable without re-fetching. The GW1 bands predate that field but chunk
   correctly, since `entries_listed == pages_requested × 50` in every one.
3. Finish `run_auto()` band iteration, timeout, and MoE reporting.
4. Verify the GW2 label pull loads the GW1 band cohorts correctly.
5. Decide on `r500k-1m` once the GW1 rank-response curve is visible.
6. ~~Split `r10k-100k` and backfill its GW1 cohorts before the GW2 deadline.~~ **Done 2026-08-27.**

---

## 9. GW1 result (backfill, 2026-08-27)

All bands captured, **0 unavailable squads**. `owned` sums to 1500.0 and `captain` to ~100.0 in
every band. Observed throughput was ~88 req/s against the 55 req/s the estimator assumes, so §5's
runtimes are conservative by roughly 40%.

> **Join on `id`, never `name`.** Ten `web_name` values are duplicated within a single band
> (Palmer, James, Wilson, Johnson, Henderson, Martinez, Davies, Gomez, Patterson, Phillips).
> Aggregating on name silently averages two different players.

The premise holds — ownership is cleanly monotonic in rank, in both directions:

| Player | top10k | 10-30k | 30-100k | 100-250k | 250-500k | 500k-1m |
|---|---|---|---|---|---|---|
| De Cuyper | 37.5 | 23.9 | 13.0 | 7.8 | 5.2 | 3.7 |
| Palmer | 53.5 | 45.5 | 36.4 | 31.8 | 24.6 | 21.9 |
| Haaland | 45.3 | 45.6 | 51.7 | 51.4 | 56.1 | 56.5 |
| Groß | 5.4 | 8.5 | 10.9 | 13.8 | 15.1 | 15.3 |

Captaincy separates the same way, and is the part invisible to ownership alone:

| Player | top10k | 10-30k | 30-100k | 100-250k | 250-500k | 500k-1m |
|---|---|---|---|---|---|---|
| João Pedro | 29.8 | 29.3 | 27.8 | 25.6 | 23.9 | 20.1 |
| Haaland | 21.8 | 23.5 | 26.3 | 28.7 | 30.6 | 35.2 |
| Palmer | 9.0 | 6.6 | 4.8 | 3.2 | 2.0 | 1.5 |

A single top-10k band would report De Cuyper at 37.5% and Haaland at 45.3% and stop there. The
slope is the feature: De Cuyper is a 10x elite differential, Haaland is mass template the elite
are *underweight*. Neither is recoverable from one band at any sample size — the case for §1.2.

### 9.1 Split validation

`r10k-30k` and `r30k-100k` were drawn independently of the `r10k-100k` pull they replace —
different pages, different managers. Their rank-width-weighted blend (20/90 and 70/90) reproduces
that band closely, which validates the sampler, page selection, aggregation and weighting
end to end:

| Metric | mean diff | mean abs diff | max abs diff | correlation |
|---|---|---|---|---|
| `owned`   | +0.000 pp | 0.099 pp | 1.30 pp | 0.99973 |
| `eo`      | +0.005 pp | 0.089 pp | 1.23 pp | 0.99977 |
| `captain` | +0.000 pp | 0.008 pp | 0.36 pp | 0.99982 |

The split earns its cost where intended: De Cuyper's decay through the steep region reads
37.5 → 23.9 → 13.0 → 7.8 instead of the previous 37.5 → 16.2 → 7.8.

### 9.2 Metadata completeness

`entry_pages` is now stored explicitly on every GW1 cohort rather than reconstructed by chunking.
This was prompted by a real failure mode: one `r30k-100k` standings page returned 49 entries
rather than 50, which breaks any positional reconstruction. Backfilling it re-fetched each band's
standings pages and verified every entry ID still matched the frozen cohort before writing —
all matched, independently confirming the cohorts are correctly captured.
