"""Compute effective ownership (EO) for a rank-band cohort of FPL managers.

EO for player i over a cohort of N managers is the mean pick multiplier:

    EO_i = sum(multiplier_i) / N

The API's `multiplier` field already encodes every chip correctly - bench 0,
starter 1, captain 2, triple captain 3, and bench boost sets all 15 to >= 1 - so
no chip adjustment is applied.

Cohorts are drawn from the Overall league (id 314), which is fully paginated at 50 entries per
page, so a rank band [lo, hi] is pages ceil(lo/50) .. ceil(hi/50).

Several disjoint bands are measured, not just the top 10k. The target is top-10k EO, but the
transfer counts that feed the model are global across all ~11m managers, and learning that
global-to-elite mapping from ~37 gameweek transitions a season is a thin basis. Measuring the
same transfer wave landing at every rank level identifies the rank response cross-sectionally
instead, and makes the shape of a player's EO-vs-rank curve available as a feature - which is
what separates an elite-led move from a mass bandwagon. See docs/eo_rank_bands.md.

Run this only once the gameweek is final. Until bonus is confirmed the standings
endpoint serves provisional totals, so an early freeze captures a cohort ranked
on numbers that then reshuffle. --force overrides the check.

Two modes, and the difference matters:

  (default)        freeze the current top-N and measure their picks for `gw`. This is the
                   t=0 state - the squads that cohort holds entering the week ahead.
  --from-cohort N  reuse the cohort frozen at gameweek N and measure *those same* managers
                   in `gw`. This is the training label: the EO that cohort actually faced.

Re-selecting the pool at the same gameweek you measure conditions the cohort on that
gameweek's results, which is why the label pull must reuse a previously frozen cohort
rather than pulling fresh standings.

Note the gates differ. A fresh pull needs the gameweek finalised and the Overall league
rebuilt, because standings are provisional until bonus is confirmed and the league tables
are recalculated later still. A --from-cohort pull needs only the deadline to have
passed, because picks are frozen at the deadline and do not depend on points. Automatic
substitutions are reported separately by the API and are deliberately ignored: the model
predicts what managers submit, not what the auto-sub engine does afterwards.
"""

import argparse
import json
import math
import os
import random
import statistics
import threading
import time
from datetime import UTC, datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from paths import DATA_DIR

API = "https://fantasy.premierleague.com/api"
OVERALL_LEAGUE_ID = 314
PAGE_SIZE = 50
OUT_DIR = DATA_DIR / "effective_ownership"
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


# Rank bands are disjoint slices of the Overall league, addressed by *positional* rank rather
# than the API's `rank` field. Early-season ties are enormous - page 20,000 currently reports
# rank 893,714 against rank_sort 999,951 - and pagination indexes position, not rank.
#
# Disjoint rather than nested (top-10k, top-100k, ...) because cumulative bands are recoverable
# from strata by size-weighted averaging, while strata are not cleanly recoverable from
# cumulative bands: subtracting two sampled estimates amplifies both their noise.
@dataclass(frozen=True)
class Band:
    name: str
    lo: int  # 1-based inclusive positional rank
    hi: int
    sample_pages: int | None  # None enumerates every page in the band

    @property
    def first_page(self) -> int:
        return (self.lo - 1) // PAGE_SIZE + 1

    @property
    def last_page(self) -> int:
        return math.ceil(self.hi / PAGE_SIZE)

    @property
    def total_pages(self) -> int:
        return self.last_page - self.first_page + 1

    @property
    def sampled(self) -> bool:
        return self.sample_pages is not None and self.sample_pages < self.total_pages


# Page budgets are weighted towards the top, where the EO-vs-rank curve actually bends. The
# target band is enumerated in full: it supplies the training label, and noise in a label
# inflates variance directly.
# BANDS is a registry of definitions, not a partition: r10k-100k is superseded by the two strata
# that split it but stays defined so its GW1 pull remains addressable and reproducible. Only
# DEFAULT_BANDS is pulled routinely, and those *are* disjoint and contiguous.
BANDS = (
    Band("top10000", 1, 10_000, None),
    Band("r10k-30k", 10_001, 30_000, 100),
    Band("r30k-100k", 30_001, 100_000, 120),
    Band("r10k-100k", 10_001, 100_000, 180),  # superseded by the two above
    Band("r100k-250k", 100_001, 250_000, 120),
    Band("r250k-500k", 250_001, 500_000, 100),
    Band("r500k-1m", 500_001, 1_000_000, 100),
)
BANDS_BY_NAME = {b.name: b for b in BANDS}

# The full default set spans rank 1 to 1m. `selected_by_percent` from bootstrap-static anchors the
# far end of the *ownership* curve for free, but it is not a substitute for the r500k-1m band: it
# carries no captaincy or bench information, and it is a live snapshot rather than a
# deadline-frozen measurement, so it is not comparable with the other bands.
DEFAULT_BANDS = ("top10000", "r10k-30k", "r30k-100k", "r100k-250k", "r250k-500k", "r500k-1m")

# Fallback only - every sampled pull now measures its own design effect (estimate_clustering).
# This is used when page labels are unavailable, e.g. a label pull whose frozen cohort predates
# the `pages` field. Set from the GW1 measurement: intra-page ICC came in at 0.005-0.008 across
# bands, well under the 0.02 originally assumed. See docs/eo_rank_bands.md section 6.
ASSUMED_ICC = 0.007


def build_session(workers: int) -> requests.Session:
    """Session with backoff on rate limiting and a connection pool sized to the worker count."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    retry = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    return session


def fetch(session: requests.Session, url: str) -> dict | None:
    """Return decoded JSON, or None if the resource is missing or unreachable."""
    try:
        response = session.get(url, timeout=30)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def fetch_many(session: requests.Session, urls: list[str], workers: int, label: str) -> list[dict | None]:
    with ThreadPoolExecutor(workers) as pool:
        return list(tqdm(pool.map(lambda u: fetch(session, u), urls), total=len(urls), desc=label, unit="req"))


def resolve_gameweek(bootstrap: dict, gw: int | None) -> int:
    """Default to the most recent finalised gameweek."""
    if gw is not None:
        return gw
    checked = [e["id"] for e in bootstrap["events"] if e["data_checked"]]
    if checked:
        return max(checked)
    current = [e["id"] for e in bootstrap["events"] if e["is_current"]]
    if not current:
        raise SystemExit("No gameweek has started yet - pass --gw explicitly.")
    return max(current)


def check_finalised(session: requests.Session, bootstrap: dict, gw: int) -> list[str]:
    """Return the reasons this gameweek is not yet safe to freeze, empty if it is."""
    event = next((e for e in bootstrap["events"] if e["id"] == gw), None)
    if event is None:
        raise SystemExit(f"Gameweek {gw} does not exist.")

    reasons = []
    if not event["finished"]:
        reasons.append("gameweek is still in progress")
    if not event["data_checked"]:
        reasons.append("data_checked is false (points not finalised)")

    status = fetch(session, f"{API}/event-status/") or {}
    pending = [d["date"] for d in status.get("status", []) if d.get("event") == gw and not d.get("bonus_added")]
    if pending:
        reasons.append(f"bonus not yet added for {', '.join(pending)}")

    # The cohort is read from the Overall league, and league tables are recalculated on their
    # own schedule - after data_checked and bonus, not with them. Freezing in that window
    # captures the *previous* gameweek's ordering while every other gate reads ready, and the
    # existing-output guard means it would never be revisited. Empty means not yet rebuilt;
    # any non-empty marker (observed as "Updated") counts as done, so an unrecognised value
    # does not block the pull forever.
    if not str(status.get("leagues") or "").strip():
        reasons.append("league tables not yet updated (standings still show the previous ranking)")
    return reasons


def tag_for(gw: int, band: Band, from_cohort: int | None) -> str:
    """Output filename stem. A label pull must not collide with the t=0 pull for the same gameweek.

    The target band is named "top10000" so its stem is unchanged from before bands existed;
    renaming it would stop the existing-output guard matching the files already on disk and
    re-pull data that is already captured.
    """
    tag = f"gw{gw}_{band.name}"
    if band.sampled:
        tag += "_sampled"
    if from_cohort is not None:
        tag += f"_cohort{from_cohort}"
    return tag


def load_cohort(from_gw: int, band: Band) -> tuple[list[int], list[int] | None, str]:
    """Entry IDs frozen at an earlier gameweek, so the same managers can be measured later.

    Also returns the page each entry was drawn from, when the frozen metadata records it. Pulls
    written before `pages` existed return None, which only means the label pull cannot report a
    design effect - it does not affect the EO numbers.
    """
    name = f"cohort_{tag_for(from_gw, band, None)}.json"
    path = OUT_DIR / name
    if not path.exists():
        raise SystemExit(f"No frozen cohort at {path}.\nRun without --from-cohort for GW{from_gw} first to create it.")
    meta = json.loads(path.read_text())
    entry_ids = meta.get("entry_ids") or []
    if not entry_ids:
        raise SystemExit(f"{path} contains no entry_ids.")
    entry_pages = meta.get("entry_pages") or None
    if entry_pages is None:
        # Pulls written before entry_pages existed: reconstruct only when the entries chunk
        # evenly, which proves no standings page came back short.
        frozen_pages = meta.get("pages") or []
        if frozen_pages and len(entry_ids) == len(frozen_pages) * PAGE_SIZE:
            entry_pages = [page for page in frozen_pages for _ in range(PAGE_SIZE)]
    if entry_pages is not None and len(entry_pages) != len(entry_ids):
        entry_pages = None
    return entry_ids, entry_pages, name


def check_deadline_passed(bootstrap: dict, gw: int) -> list[str]:
    """Picks are fixed at the deadline, so a label pull does not need the gameweek finalised."""
    event = next((e for e in bootstrap["events"] if e["id"] == gw), None)
    if event is None:
        raise SystemExit(f"Gameweek {gw} does not exist.")
    if datetime.now(UTC) < datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00")):
        return [f"the GW{gw} deadline has not passed yet ({event['deadline_time']})"]
    return []


def select_pages(band: Band, seed: int) -> list[int]:
    """Pages covering the band, thinned systematically when the band is sampled.

    Systematic rather than a uniform random draw: 100 pages drawn at random from 10,000 leave
    large rank gaps by chance, and the point of a band is to characterise EO across its whole
    rank range.

    The offset is seeded on the band name and deliberately *not* on the gameweek, so the same
    rank slices are revisited every week. The model predicts week-over-week changes in EO, and
    holding the slices fixed cancels most of the cluster noise from those deltas; redrawing each
    week would add independent noise to every one. This fixes the rank slices, not the managers -
    following the same managers is the separate axis handled by --from-cohort.
    """
    if not band.sampled:
        return list(range(band.first_page, band.last_page + 1))
    step = band.total_pages / band.sample_pages
    offset = random.Random(f"{seed}:{band.name}").random() * step
    pages = {band.first_page + int(i * step + offset) for i in range(band.sample_pages)}
    return sorted(p for p in pages if p <= band.last_page)


def fetch_cohort(session: requests.Session, pages: list[int], workers: int) -> tuple[list[int], list[int]]:
    """Entry IDs for the requested pages, plus the page each one came from.

    The page labels are the cluster identifiers behind the design-effect estimate. Building them
    alongside the IDs rather than chunking the flat list afterwards keeps them correct when a
    standings page fails and its 50 entries are missing.
    """
    urls = [f"{API}/leagues-classic/{OVERALL_LEAGUE_ID}/standings/?page_standings={p}" for p in pages]
    responses = fetch_many(session, urls, workers, "standings")

    entry_ids: list[int] = []
    entry_pages: list[int] = []
    failed = 0
    for page, response in zip(pages, responses):
        if not response or "standings" not in response:
            failed += 1
            continue
        results = response["standings"]["results"]
        entry_ids.extend(r["entry"] for r in results)
        entry_pages.extend([page] * len(results))
    if failed:
        print(f"  warning: {failed} standings pages failed and were skipped")
    return entry_ids, entry_pages


def aggregate(squads: list[dict | None], entry_pages: list[int] | None = None) -> tuple[dict[str, Counter], int, Counter, dict[int, Counter], Counter]:
    """Sum multipliers and role counts across every squad that was retrieved.

    When entry_pages is supplied (one page label per squad, same order), ownership is also
    tallied per page. Those per-page counts are what estimate_clustering turns into a design
    effect - computed here because the picks are already in memory, so it costs no extra
    requests to know how precise the sample actually is.
    """
    tallies = {k: Counter() for k in ("eo", "owned", "started", "benched", "captain", "vice", "triple")}
    page_owned: dict[int, Counter] = defaultdict(Counter)
    page_n: Counter = Counter()
    chips = Counter()
    sampled = 0

    for index, squad in enumerate(squads):
        if not squad or "picks" not in squad:
            continue
        sampled += 1
        chips[squad.get("active_chip") or "none"] += 1
        page = entry_pages[index] if entry_pages is not None and index < len(entry_pages) else None
        if page is not None:
            page_n[page] += 1

        for pick in squad["picks"]:
            element = pick["element"]
            multiplier = pick["multiplier"]
            tallies["eo"][element] += multiplier
            tallies["owned"][element] += 1
            if multiplier > 0:
                tallies["started"][element] += 1
            else:
                tallies["benched"][element] += 1
            if pick["is_captain"]:
                tallies["captain"][element] += 1
                if multiplier == 3:
                    tallies["triple"][element] += 1
            if pick["is_vice_captain"]:
                tallies["vice"][element] += 1
            if page is not None:
                page_owned[page][element] += 1
    return tallies, sampled, chips, dict(page_owned), page_n


def estimate_clustering(page_owned: dict[int, Counter], page_n: Counter, min_rate: float = 0.02, max_rate: float = 0.98) -> dict | None:
    """Design effect of the page-cluster sample, measured rather than assumed.

    Each page is PAGE_SIZE managers of adjacent rank taken as a block, so they are not
    PAGE_SIZE independent draws. For one player, the ownership rate on each sampled page would
    scatter around the band mean with variance p(1-p)/m if they were; any excess is the
    page-to-page effect. Two estimators, because they answer different questions:

      deff_cluster    the textbook ratio, treating pages as exchangeable clusters. Yields the
                      ICC that DEFF = 1 + (m-1)*ICC is defined against.
      deff_systematic the successive-difference estimator, which uses the variance between
                      *adjacent* sampled pages. This is the honest one for our design: page
                      order is rank order and select_pages spreads the sample evenly across the
                      band, so a smooth ownership-vs-rank gradient is sampled like a stratified
                      design and costs far less precision than the exchangeable-cluster formula
                      charges for it. Expect it to be the smaller of the two.

    The median across players is reported rather than the mean: the per-player ratios are
    heavy-tailed, and a handful of near-template or near-zero players would otherwise dominate.
    """
    pages = sorted(p for p in page_n if page_n[p] > 0)
    k = len(pages)
    if k < 8:
        return None
    m = sum(page_n[p] for p in pages) / k

    ratios = []
    for element in {e for p in pages for e in page_owned.get(p, ())}:
        rates = [page_owned.get(p, {}).get(element, 0) / page_n[p] for p in pages]
        mean = sum(rates) / k
        if not min_rate <= mean <= max_rate:
            continue
        var_independent = mean * (1 - mean) / (k * m)
        var_cluster = sum((r - mean) ** 2 for r in rates) / (k - 1) / k
        var_systematic = sum((rates[i] - rates[i - 1]) ** 2 for i in range(1, k)) / (2 * k * (k - 1))
        ratios.append((var_cluster / var_independent, var_systematic / var_independent))
    if not ratios:
        return None

    deff = statistics.median(r[0] for r in ratios)
    deff_sys = statistics.median(r[1] for r in ratios)
    return {
        "players_used": len(ratios),
        "pages": k,
        "mean_page_n": round(m, 1),
        "deff_cluster": round(deff, 3),
        "icc_cluster": round((deff - 1) / (m - 1), 5),
        "deff_systematic": round(deff_sys, 3),
        "icc_systematic": round((deff_sys - 1) / (m - 1), 5),
    }


def build_table(bootstrap: dict, tallies: dict[str, Counter], sampled: int, band: Band) -> pd.DataFrame:
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    rows = []

    for element in bootstrap["elements"]:
        element_id = element["id"]
        if tallies["owned"][element_id] == 0:
            continue
        rows.append(
            {
                "band": band.name,
                "id": element_id,
                "name": element["web_name"],
                "team": teams.get(element["team"], "?"),
                "pos": POSITIONS.get(element["element_type"], "?"),
                "price": element["now_cost"] / 10,
                "eo": 100 * tallies["eo"][element_id] / sampled,
                "owned": 100 * tallies["owned"][element_id] / sampled,
                "started": 100 * tallies["started"][element_id] / sampled,
                "benched": 100 * tallies["benched"][element_id] / sampled,
                "captain": 100 * tallies["captain"][element_id] / sampled,
                "vice": 100 * tallies["vice"][element_id] / sampled,
                "triple_captain": 100 * tallies["triple"][element_id] / sampled,
                "global_owned": float(element["selected_by_percent"]),
            }
        )

    frame = pd.DataFrame(rows)
    frame["ownership_delta"] = frame["owned"] - frame["global_owned"]
    return frame.sort_values("eo", ascending=False).round(2).reset_index(drop=True)


def resolve_bands(args: argparse.Namespace) -> list[Band]:
    """Bands named on the command line, or the default set."""
    if not args.bands:
        return [BANDS_BY_NAME[n] for n in DEFAULT_BANDS]
    names = [n.strip() for n in args.bands.split(",") if n.strip()]
    unknown = [n for n in names if n not in BANDS_BY_NAME]
    if unknown:
        raise SystemExit(f"Unknown band(s): {', '.join(unknown)}.\nAvailable: {', '.join(BANDS_BY_NAME)}")
    return [BANDS_BY_NAME[n] for n in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute effective ownership by rank band from the FPL API.")
    parser.add_argument("--gw", type=int, default=None, help="gameweek to measure (default: most recent finalised)")
    parser.add_argument("--bands", default=None, metavar="NAMES", help=f"comma-separated rank bands (default: {','.join(DEFAULT_BANDS)}; available: {','.join(BANDS_BY_NAME)})")
    parser.add_argument("--workers", type=int, default=24, help="concurrent requests (default: 24)")
    parser.add_argument("--seed", type=int, default=0, help="seed for page sampling (default: 0)")
    parser.add_argument("--top", type=int, default=30, help="rows to print (default: 30)")
    parser.add_argument("--from-cohort", type=int, metavar="GW", default=None, help="reuse the cohort frozen at that gameweek instead of pulling fresh standings (label pull)")
    parser.add_argument("--timeout", type=float, default=150.0, help="abort after this many minutes, 0 to disable (default: 150)")
    parser.add_argument("--auto", action="store_true", help="do whichever pulls are possible and not yet captured (for scheduled runs)")
    parser.add_argument("--auto-window", type=int, default=3, help="with --auto, how many recent gameweeks to consider (default: 3)")
    parser.add_argument("--overwrite", action="store_true", help="redo a pull whose output already exists")
    parser.add_argument("--force", action="store_true", help="run even if the gameweek is not finalised")
    return parser.parse_args()


def arm_timeout(minutes: float) -> None:
    """Hard wall-clock cap. launchd imposes no runtime limit, so an unattended run that
    stalls - a degraded API plus retries across thousands of requests - would otherwise sit
    there indefinitely. Exits 2 so a scheduled run reports failure rather than hanging."""
    if minutes <= 0:
        return

    def bail() -> None:
        print(f"\nTIMEOUT: exceeded {minutes:g} min - aborting.", flush=True)
        os._exit(2)

    timer = threading.Timer(minutes * 60, bail)
    timer.daemon = True
    timer.start()


def run_pull(session: requests.Session, bootstrap: dict, args: argparse.Namespace, gw: int, band: Band, from_cohort: int | None, strict: bool = True) -> bool:
    """One EO pull for one rank band. Returns True if it wrote output, False if it was skipped.

    With strict=False a closed gate is a skip rather than an error, which is what --auto
    needs: it walks recent gameweeks and does whatever is currently possible.
    """
    label_pull = from_cohort is not None
    tag = tag_for(gw, band, from_cohort)
    where = f"GW{gw} {band.name}{f' (cohort {from_cohort})' if label_pull else ''}"

    # The guard: identical inputs give identical output, so re-running daily would burn
    # thousands of requests to rewrite the same file.
    out_path = OUT_DIR / f"eo_{tag}.csv"
    if out_path.exists() and not args.overwrite:
        print(f"{where}: already captured at {out_path.name} - skipping. Use --overwrite to redo.")
        return False

    if label_pull:
        reasons = check_deadline_passed(bootstrap, gw)
        advice = "Picks only exist once the deadline has passed."
    else:
        reasons = check_finalised(session, bootstrap, gw)
        advice = "Standings are provisional until bonus is confirmed."
    if reasons:
        message = f"Gameweek {gw}: {'; '.join(reasons)}."
        if args.force:
            print(f"WARNING: {message} Results are provisional.\n")
        elif strict:
            raise SystemExit(f"{message}\n{advice} Re-run later, or pass --force.")
        else:
            print(f"{where}: not ready - {'; '.join(reasons)}")
            return False

    start = time.time()
    if label_pull:
        pages = []
        try:
            entry_ids, entry_pages, cohort_source = load_cohort(from_cohort, band)
        except SystemExit:
            if strict:
                raise
            print(f"{where}: no frozen GW{from_cohort} {band.name} cohort yet - skipping label pull")
            return False
        print(f"Gameweek {gw} | band {band.name} (ranks {band.lo:,}-{band.hi:,}) | cohort frozen at GW{from_cohort} ({cohort_source}) | {len(entry_ids):,} managers")
        print(f"Estimated runtime ~{len(entry_ids) / (2.3 * args.workers) / 60:.1f} min\n")
    else:
        pages = select_pages(band, args.seed)
        cohort_source = None
        expected = len(pages) * PAGE_SIZE
        coverage = f"{len(pages):,} of {band.total_pages:,} pages sampled" if band.sampled else f"all {len(pages):,} pages"
        print(f"Gameweek {gw} | band {band.name} (ranks {band.lo:,}-{band.hi:,}) | {coverage} -> ~{expected:,} managers")
        print(f"Estimated runtime ~{(len(pages) + expected) / (2.3 * args.workers) / 60:.1f} min\n")
        entry_ids, entry_pages = fetch_cohort(session, pages, args.workers)

    squads = fetch_many(session, [f"{API}/entry/{e}/event/{gw}/picks/" for e in entry_ids], args.workers, "picks")
    tallies, sampled, chips, page_owned, page_n = aggregate(squads, entry_pages)
    if sampled == 0:
        message = "No squads retrieved - the gameweek may predate the cohort's entries."
        if strict:
            raise SystemExit(message)
        # Under --auto this must not abort the run: the remaining gameweeks in the window
        # are independent, and a transient API failure here would otherwise cost them too.
        print(f"{where}: {message}")
        return False

    clustering = estimate_clustering(page_owned, page_n) if band.sampled else None
    table = build_table(bootstrap, tallies, sampled, band)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    meta = {
        "gameweek": gw,
        "band": band.name,
        "rank_lo": band.lo,
        "rank_hi": band.hi,
        "pages_requested": len(pages),
        # The actual page numbers, so intra-page variance (the design effect behind ASSUMED_ICC)
        # is computable from the artefacts without re-fetching. entry_ids below are stored in
        # page order, PAGE_SIZE per page, so the two zip together whenever no page failed -
        # i.e. whenever entries_listed == pages_requested * PAGE_SIZE.
        "pages": pages,
        "band_total_pages": band.total_pages,
        "entries_listed": len(entry_ids),
        "squads_sampled": sampled,
        "sampled_pages": band.sampled,
        "sample_pages": band.sample_pages,
        "seed": args.seed,
        "forced": args.force,
        "chips": dict(chips),
        # UTC with an explicit marker. time.strftime writes naive local time, which reads as
        # a different instant depending on where and when it was produced - Melbourne shifts
        # by an hour at the October DST change - and nothing downstream could detect it.
        "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_source": cohort_source,
        # Measured design effect, or None when the band is unsampled or page labels are absent.
        "clustering": clustering,
        # Only a fresh pull freezes a cohort; a label pull points back at the one it reused.
        # entry_pages is stored rather than reconstructed by chunking: a standings page can
        # return fewer than PAGE_SIZE entries (observed at GW1), which silently breaks any
        # positional reconstruction and would cost the label pull its design effect.
        "entry_ids": [] if label_pull else entry_ids,
        "entry_pages": [] if label_pull else (entry_pages or []),
    }
    (OUT_DIR / f"cohort_{tag}.json").write_text(json.dumps(meta))

    print(f"\nSampled {sampled:,} squads ({len(entry_ids) - sampled:,} unavailable) in {time.time() - start:.0f}s")
    print(f"Chips: {', '.join(f'{k} {100 * v / sampled:.1f}%' for k, v in chips.most_common())}")
    if band.sampled:
        # Each page is PAGE_SIZE adjacent ranks, so this is a cluster sample and the
        # independent-sample formula overstates precision. Prefer the measured design effect;
        # ASSUMED_ICC is only the fallback when page labels were unavailable.
        if clustering:
            deff = clustering["deff_systematic"]
            source = f"measured, systematic (exchangeable-cluster reading {clustering['deff_cluster']:.2f}, ICC {clustering['icc_cluster']:.4f})"
        else:
            deff = 1 + (PAGE_SIZE - 1) * ASSUMED_ICC
            source = f"assumed ICC {ASSUMED_ICC}"
        n_eff = sampled / max(deff, 1e-9)
        print(f"Clustered sample - design effect {deff:.2f} ({source}), effective n ~{n_eff:,.0f}")
        print(f"95% margin of error at most +/-{100 * 1.96 * math.sqrt(0.25 / n_eff):.2f} pts")
    if args.top:
        print(f"\nTop {args.top} by EO, gameweek {gw}, band {band.name} (ranks {band.lo:,}-{band.hi:,}):\n")
        print(table.head(args.top).to_string(index=False))
    print(f"\nWritten to {out_path}")
    if label_pull:
        print(f"Label for GW{gw} measured against the GW{from_cohort} {band.name} cohort")
    else:
        print(f"Cohort IDs frozen in {OUT_DIR / f'cohort_{tag}.json'}")
    return True


def run_auto(session: requests.Session, bootstrap: dict, args: argparse.Namespace) -> None:
    """Do whichever pulls are currently possible and not already captured.

    Each gameweek needs two pulls per band, gated differently: the label (previous cohort
    measured in this gameweek) becomes available once the deadline passes, the t=0 state once the
    gameweek finalises. Running this daily lands each one shortly after it becomes possible,
    and does nothing the rest of the time.
    """
    now = datetime.now(UTC)
    passed = [e for e in bootstrap["events"] if datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")) <= now]
    if not passed:
        print("No gameweek deadline has passed yet - nothing to do.")
        return

    bands = resolve_bands(args)
    recent = sorted(passed, key=lambda e: e["id"])[-args.auto_window :]
    print(f"Auto mode: checking GW{recent[0]['id']}-{recent[-1]['id']} over {len(bands)} band(s): {', '.join(b.name for b in bands)}\n")

    written = 0
    for event in recent:
        gw = event["id"]
        for band in bands:
            if gw > 1:
                written += run_pull(session, bootstrap, args, gw, band, gw - 1, strict=False)
            written += run_pull(session, bootstrap, args, gw, band, None, strict=False)
            print()
    print(f"Auto mode complete: {written} pull(s) written.")


def main() -> None:
    args = parse_args()
    arm_timeout(args.timeout)
    session = build_session(args.workers)

    bootstrap = fetch(session, f"{API}/bootstrap-static/")
    if bootstrap is None:
        raise SystemExit("Could not reach the FPL API.")

    if args.auto:
        run_auto(session, bootstrap, args)
    else:
        gw = resolve_gameweek(bootstrap, args.gw)
        for band in resolve_bands(args):
            run_pull(session, bootstrap, args, gw, band, args.from_cohort)


if __name__ == "__main__":
    main()
