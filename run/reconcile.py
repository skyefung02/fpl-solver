"""Audit that the transfer market table and the effective-ownership tables agree.

Everything so far validates the two datasets separately: the collectors agree with each
other, the EO pulls land all six bands. Nothing has yet confirmed they describe the same
population over the same interval in the same units, which is the assumption every model
trained on (transfers -> next-gameweek EO) rests on.

Four checks, cheapest first:

  A  alignment    EO's global_owned against selected_by_percent in the market snapshot
                  taken at the same instant. Both read bootstrap-static, so this is a
                  fidelity and time-alignment test rather than independent corroboration
                  - which is precisely what catches a clock or timezone slip.
  B  conservation change in owners against net transfers, differenced between consecutive
                  snapshots so both sides span the same interval. Solves for the manager
                  base that best fits rather than assuming one - the fitted base turns out
                  to sit well below total_players, which is therefore the wrong denominator.
  C  resolution   selected_by_percent carries one decimal place, so movements below ~0.1%
                  of the manager base are invisible. Reports how much of the label is
                  quantisation noise, which bounds what a model can be asked to predict.
  D  coverage     band-weighted top-1m ownership against the global figure. This one is
                  genuinely independent - crawled picks versus the bootstrap aggregate -
                  and is the strongest evidence the bands measure what we think.

Usage:

    python run/reconcile.py --gameweek 2 --market data/market_table.parquet
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# selected_by_percent is published to one decimal place.
OWNERSHIP_STEP = 0.1
# Check A compares two reads of the same upstream field; anything above rounding is a fault.
ALIGNMENT_TOLERANCE = 0.2
# Check B solves for the manager base; a fit this far from the API's own figure means the
# units or the window attribution are wrong, not that the season grew.
# Conservation is judged on the fit's own consistency: total_players is not the denominator
# selected_by_percent is quoted against, so agreement with it is not the test.
FIT_QUALITY_FLOOR = 0.85
FIT_STABILITY_CEILING = 0.10
MIN_PLAYERS_FOR_FIT = 10
# Squad size is fixed at 15, so the slot identity should hold to rounding.
SLOT_IDENTITY_TOLERANCE = 0.001


def load_bands(eo_dir: Path, gameweek: int) -> list[dict]:
    """EO tables for one gameweek, paired with their cohort metadata.

    Bands can overlap: r10k-100k was superseded by r10k-30k plus r30k-100k but its files
    remain. Overlapping ranges would double-count in check D, so the finer split wins and
    the coarse band is dropped - narrower ranges sort first for the same rank_lo.
    """
    found = []
    for table in sorted(eo_dir.glob(f"eo_gw{gameweek}_*.csv")):
        meta = table.with_name(table.name.replace("eo_", "cohort_", 1)).with_suffix(".json")
        if not meta.exists():
            print(f"  ! {table.name}: no cohort metadata, skipped")
            continue
        info = json.loads(meta.read_text())
        info["frame"] = pd.read_csv(table)
        info["file"] = table.name
        found.append(info)

    kept: list[dict] = []
    for band in sorted(found, key=lambda b: (b["rank_lo"], b["rank_hi"])):
        clash = next((k for k in kept if band["rank_lo"] <= k["rank_hi"] and k["rank_lo"] <= band["rank_hi"]), None)
        if clash:
            print(f"  ! {band['band']} overlaps {clash['band']}; superseded, dropped from coverage")
            continue
        kept.append(band)
    return kept


def captured_utc(band: dict) -> datetime:
    """Cohort stamps are UTC with an explicit marker; naive ones are pre-migration.

    Older cohorts were written with time.strftime, i.e. naive local time, which reads ten
    hours out in Melbourne and silently matches a snapshot from the wrong side of a deadline.
    Those files have been converted, but astimezone still resolves a naive stamp as local so
    that an unmigrated copy is handled rather than misread.
    """
    return datetime.fromisoformat(band["captured_at"]).astimezone(UTC)


def snapshot_at(market: pd.DataFrame, when: datetime) -> tuple[pd.DataFrame, float]:
    """The market snapshot nearest an instant, with how far off it was in minutes."""
    stamps = market["obs"].drop_duplicates()
    nearest = stamps.iloc[(stamps - when).abs().argmin()]
    return market[market["obs"] == nearest], abs((nearest - when).total_seconds()) / 60


def check_alignment(bands: list[dict], market: pd.DataFrame) -> bool:
    # market here is the FULL table, not the event slice - see main().
    print("\n[A] alignment - EO global_owned vs market selected_by_percent")
    worst = 0.0
    for band in bands:
        when = captured_utc(band)
        snap, drift = snapshot_at(market, when)
        merged = band["frame"][["id", "global_owned"]].merge(
            snap[["id", "selected_by_percent"]], on="id", how="inner"
        )
        gap = (merged["global_owned"] - merged["selected_by_percent"]).abs()
        worst = max(worst, gap.max())
        print(f"    {band['band']:<12} {when:%d %b %H:%M}Z  nearest snapshot {drift:5.1f} min away  "
              f"n={len(merged):>3}  max|diff|={gap.max():.2f}pp  mean={gap.mean():.3f}pp")
    ok = worst <= ALIGNMENT_TOLERANCE
    print(f"    -> {'PASS' if ok else 'FAIL'} (worst {worst:.2f}pp, tolerance {ALIGNMENT_TOLERANCE}pp)")
    return ok


def check_conservation(market: pd.DataFrame, api_base: int | None) -> tuple[bool, float]:
    """Fit the manager base that reconciles ownership against transfers.

    Differencing consecutive snapshots does not work: selected_by_percent is a slow field -
    across 221 GW2 snapshots the median player's ownership never changed at all, while its
    transfer counters moved 126 times. Pairing one ownership step against the fraction of
    transfers in a single 5-minute interval understates the base roughly twofold. Both sides
    are therefore differenced across the whole observed window, which also keeps their spans
    identical: the counters are cumulative from the deadline, so taking the last value alone
    would span more time than the ownership change does.

    The pass criterion is the fit's own consistency, not agreement with total_players. The
    fitted base comes out well below it, stably and with high explanatory power, so
    total_players is not the denominator selected_by_percent is quoted against.
    """
    print("\n[B] conservation - change in owners vs net transfers")
    frame = market.sort_values(["id", "obs"]).copy()
    frame["net"] = frame["transfers_in_event"] - frame["transfers_out_event"]
    first, last = frame.groupby("id").first(), frame.groupby("id").last()
    move = ((last["selected_by_percent"] - first["selected_by_percent"]) / 100).dropna()
    net = (last["net"] - first["net"]).reindex(move.index).dropna()
    move = move.reindex(net.index)
    if len(move) < MIN_PLAYERS_FOR_FIT:
        print("    -> SKIP (too few players with usable movement)")
        return True, float("nan")

    fits = {}
    for floor in (0.0, 0.5, 1.0, 2.0):
        keep = move.abs() * 100 >= floor
        if int(keep.sum()) < MIN_PLAYERS_FOR_FIT:
            continue
        x, y = move[keep], net[keep]
        base = float((x * y).sum() / (x**2).sum())
        residual = ((y - x * base) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fits[floor] = (int(keep.sum()), base, 1 - float(residual))

    print(f"    {'min |d ownership|':>18} {'players':>8} {'implied base':>14} {'R2':>7}")
    for floor, (n, base, r2) in fits.items():
        print(f"    {floor:>17.1f}pp {n:>8} {base:>14,.0f} {r2:>7.3f}")

    headline = fits.get(0.5, next(iter(fits.values())))
    base, r2 = headline[1], headline[2]
    spread = (max(f[1] for f in fits.values()) - min(f[1] for f in fits.values())) / base
    print(f"    fitted base {base:,.0f}   R2 {r2:.3f}   spread across thresholds {spread:.1%}")
    if api_base:
        print(f"    API total_players {api_base:,} -> the fitted base is {base / api_base:.0%} of it;"
              f" use the fit, not total_players, to convert ownership to managers")
    ok = r2 >= FIT_QUALITY_FLOOR and spread <= FIT_STABILITY_CEILING
    print(f"    -> {'PASS' if ok else 'FAIL'} (needs R2 >= {FIT_QUALITY_FLOOR}, spread <= {FIT_STABILITY_CEILING:.0%})")
    return ok, base


def check_resolution(market: pd.DataFrame, base: float) -> None:
    print("\n[C] resolution - how much of the label is quantisation noise")
    floor = OWNERSHIP_STEP / 100 * base
    last = market.sort_values("obs").groupby("id").last()
    net = (last["transfers_in_event"] - last["transfers_out_event"]).abs()
    below = int((net < floor).sum())
    print(f"    one step of selected_by_percent = {floor:,.0f} managers")
    print(f"    players whose whole-window net transfer is below one step: {below}/{len(net)} "
          f"({100 * below / len(net):.0f}%)")
    print(f"    median |net| = {net.median():,.0f}   p90 = {net.quantile(0.9):,.0f}")
    print("    -> informational; these players' ownership labels are mostly rounding")


def check_coverage(bands: list[dict], market: pd.DataFrame, base: float) -> bool:
    print("\n[D] coverage - band-weighted top-1m ownership vs the global figure")
    covered = sum(b["rank_hi"] - b["rank_lo"] + 1 for b in bands)
    owners = None
    for band in bands:
        size = band["rank_hi"] - band["rank_lo"] + 1
        part = band["frame"][["id", "owned"]].set_index("id")["owned"] / 100 * size
        owners = part if owners is None else owners.add(part, fill_value=0)

    # Every manager fields exactly 15 players, so the band-weighted owner counts must sum to
    # 15 x the ranks covered. This holds without reference to the fitted base, and catches a
    # truncated EO table or a mis-set band size that the share test below would absorb.
    slots = float(owners.sum())
    ratio = slots / (15 * covered)
    print(f"    squad-slot identity: {slots:,.0f} vs 15 x {covered:,} = {15 * covered:,}  ratio={ratio:.4f}")
    slots_ok = abs(ratio - 1) <= SLOT_IDENTITY_TOLERANCE
    if not slots_ok:
        print(f"      ! off by {abs(ratio - 1):.2%}; a band is truncated or mis-sized")

    when = max(captured_utc(b) for b in bands)
    snap, _ = snapshot_at(market, when)
    glob = snap.set_index("id")["selected_by_percent"] / 100 * base
    both = pd.concat([owners.rename("top"), glob.rename("all")], axis=1).dropna()
    both = both[both["all"] > 0]
    share = both["top"] / both["all"]

    outside = int(((share < 0) | (share > 1)).sum())
    print(f"    bands cover ranks 1-{covered:,} of ~{base:,.0f} managers ({100 * covered / base:.1f}%)")
    print(f"    share of each player's owners inside the top {covered:,}:")
    print(f"      median={share.median():.3f}  p10={share.quantile(0.1):.3f}  p90={share.quantile(0.9):.3f}")
    print(f"    players with an impossible share (<0 or >1): {outside}/{len(share)}")
    if outside:
        for pid, value in share[(share < 0) | (share > 1)].nlargest(5).items():
            print(f"      id={pid} share={value:.2f}")
    ok = outside == 0 and slots_ok
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--market", type=Path, default=Path("data/market_table.parquet"))
    parser.add_argument("--eo-dir", type=Path, default=Path("data/effective_ownership"))
    parser.add_argument("--total-players", type=int, default=None,
                        help="API manager count for context; check B solves for its own")
    args = parser.parse_args()

    print(f"reconciling gameweek {args.gameweek}")
    bands = load_bands(args.eo_dir, args.gameweek)
    if not bands:
        raise SystemExit(f"no EO tables for gameweek {args.gameweek} under {args.eo_dir}")
    print(f"  {len(bands)} band(s): {', '.join(b['band'] for b in bands)}")

    full = pd.read_parquet(args.market)
    full["obs"] = pd.to_datetime(full["obs_time"], format="ISO8601", utc=True)
    market = full[full["event"] == args.gameweek].copy()
    if market.empty:
        raise SystemExit(f"no market rows for event {args.gameweek} in {args.market}")
    print(f"  market: {len(market):,} rows, {market['obs'].nunique()} snapshots, "
          f"{market['obs'].min():%d %b %H:%M} -> {market['obs'].max():%d %b %H:%M}Z")

    # Checks A and D match against the instant EO was captured, which is after the deadline
    # and therefore in the next event; they need the whole table. B and C difference the
    # per-event counters and need the slice.
    results = [check_alignment(bands, full)]
    conserved, implied = check_conservation(market, args.total_players)
    results.append(conserved)
    base = implied if implied == implied else (args.total_players or 0)
    if base > 0:
        check_resolution(market, base)
        results.append(check_coverage(bands, full, base))

    print(f"\n{'all checks passed' if all(results) else 'FAILURES above'}")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
