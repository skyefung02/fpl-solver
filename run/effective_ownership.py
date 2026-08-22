"""Compute effective ownership (EO) for a rank-band cohort of FPL managers.

EO for player i over a cohort of N managers is the mean pick multiplier:

    EO_i = sum(multiplier_i) / N

The API's `multiplier` field already encodes every chip correctly - bench 0,
starter 1, captain 2, triple captain 3, and bench boost sets all 15 to >= 1 - so
no chip adjustment is applied.

The cohort is drawn from the Overall league (id 314), which is fully paginated at
50 entries per page, so rank band [1, pool] is pages 1 .. pool/50.

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
import threading
import time
from datetime import UTC, datetime
from collections import Counter
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
DEFAULT_POOL = 10_000
WIDE_POOL = 100_000
WIDE_POOL_PAGES = 100


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


def tag_for(gw: int, pool: int, sample_pages: int | None, from_cohort: int | None) -> str:
    """Output filename stem. A label pull must not collide with the t=0 pull for the same gameweek."""
    tag = f"gw{gw}_top{pool}"
    if sample_pages is not None:
        tag += "_sampled"
    if from_cohort is not None:
        tag += f"_cohort{from_cohort}"
    return tag


def load_cohort(from_gw: int, pool: int, sample_pages: int | None) -> tuple[list[int], str]:
    """Entry IDs frozen at an earlier gameweek, so the same managers can be measured later."""
    name = f"cohort_{tag_for(from_gw, pool, sample_pages, None)}.json"
    path = OUT_DIR / name
    if not path.exists():
        raise SystemExit(f"No frozen cohort at {path}.\nRun without --from-cohort for GW{from_gw} first to create it.")
    entry_ids = json.loads(path.read_text()).get("entry_ids") or []
    if not entry_ids:
        raise SystemExit(f"{path} contains no entry_ids.")
    return entry_ids, name


def check_deadline_passed(bootstrap: dict, gw: int) -> list[str]:
    """Picks are fixed at the deadline, so a label pull does not need the gameweek finalised."""
    event = next((e for e in bootstrap["events"] if e["id"] == gw), None)
    if event is None:
        raise SystemExit(f"Gameweek {gw} does not exist.")
    if datetime.now(UTC) < datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00")):
        return [f"the GW{gw} deadline has not passed yet ({event['deadline_time']})"]
    return []


def select_pages(pool: int, sample_pages: int | None, seed: int) -> list[int]:
    """Full page range for the rank band, or a random subset of pages when sampling."""
    total_pages = math.ceil(pool / PAGE_SIZE)
    if sample_pages is None or sample_pages >= total_pages:
        return list(range(1, total_pages + 1))
    return sorted(random.Random(seed).sample(range(1, total_pages + 1), sample_pages))


def fetch_cohort(session: requests.Session, pages: list[int], workers: int) -> list[int]:
    urls = [f"{API}/leagues-classic/{OVERALL_LEAGUE_ID}/standings/?page_standings={p}" for p in pages]
    responses = fetch_many(session, urls, workers, "standings")

    entry_ids = []
    failed = 0
    for response in responses:
        if not response or "standings" not in response:
            failed += 1
            continue
        entry_ids.extend(r["entry"] for r in response["standings"]["results"])
    if failed:
        print(f"  warning: {failed} standings pages failed and were skipped")
    return entry_ids


def aggregate(squads: list[dict | None]) -> tuple[dict[str, Counter], int, Counter]:
    """Sum multipliers and role counts across every squad that was retrieved."""
    tallies = {k: Counter() for k in ("eo", "owned", "started", "benched", "captain", "vice", "triple")}
    chips = Counter()
    sampled = 0

    for squad in squads:
        if not squad or "picks" not in squad:
            continue
        sampled += 1
        chips[squad.get("active_chip") or "none"] += 1

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
    return tallies, sampled, chips


def build_table(bootstrap: dict, tallies: dict[str, Counter], sampled: int) -> pd.DataFrame:
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    rows = []

    for element in bootstrap["elements"]:
        element_id = element["id"]
        if tallies["owned"][element_id] == 0:
            continue
        rows.append(
            {
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


def resolve_scope(args: argparse.Namespace) -> tuple[int, int | None]:
    """Rank band and page sampling, with --100k as shorthand for a sampled top-100k run."""
    pool = args.pool if args.pool is not None else (WIDE_POOL if args.wide else DEFAULT_POOL)
    sample_pages = args.sample_pages
    if sample_pages is None and args.wide:
        sample_pages = WIDE_POOL_PAGES
    return pool, sample_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute top-N effective ownership from the FPL API.")
    parser.add_argument("--gw", type=int, default=None, help="gameweek to measure (default: most recent finalised)")
    parser.add_argument("--100k", dest="wide", action="store_true", help=f"top {WIDE_POOL:,} instead, sampled over {WIDE_POOL_PAGES} pages")
    parser.add_argument("--pool", type=int, default=None, help=f"override the rank band (default: {DEFAULT_POOL:,}, or {WIDE_POOL:,} with --100k)")
    parser.add_argument("--sample-pages", type=int, default=None, help="sample this many standings pages instead of enumerating the full band")
    parser.add_argument("--workers", type=int, default=24, help="concurrent requests (default: 24)")
    parser.add_argument("--seed", type=int, default=0, help="seed for page sampling (default: 0)")
    parser.add_argument("--top", type=int, default=30, help="rows to print (default: 30)")
    parser.add_argument("--from-cohort", type=int, metavar="GW", default=None, help="reuse the cohort frozen at that gameweek instead of pulling fresh standings (label pull)")
    parser.add_argument("--timeout", type=float, default=45.0, help="abort after this many minutes, 0 to disable (default: 45)")
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


def run_pull(session: requests.Session, bootstrap: dict, args: argparse.Namespace, gw: int, from_cohort: int | None, strict: bool = True) -> bool:
    """One EO pull. Returns True if it wrote output, False if it was skipped.

    With strict=False a closed gate is a skip rather than an error, which is what --auto
    needs: it walks recent gameweeks and does whatever is currently possible.
    """
    pool, sample_pages = resolve_scope(args)
    label_pull = from_cohort is not None
    tag = tag_for(gw, pool, sample_pages, from_cohort)

    # The guard: identical inputs give identical output, so re-running daily would burn
    # thousands of requests to rewrite the same file.
    out_path = OUT_DIR / f"eo_{tag}.csv"
    if out_path.exists() and not args.overwrite:
        print(f"GW{gw}{f' (cohort {from_cohort})' if label_pull else ''}: already captured at {out_path.name} - skipping. Use --overwrite to redo.")
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
            print(f"GW{gw}{f' (cohort {from_cohort})' if label_pull else ''}: not ready - {'; '.join(reasons)}")
            return False

    start = time.time()
    if label_pull:
        pages = []
        try:
            entry_ids, cohort_source = load_cohort(from_cohort, pool, sample_pages)
        except SystemExit:
            if strict:
                raise
            print(f"GW{gw}: no frozen GW{from_cohort} cohort yet - skipping label pull")
            return False
        print(f"Gameweek {gw} | cohort frozen at GW{from_cohort} ({cohort_source}) | {len(entry_ids):,} managers")
        print(f"Estimated runtime ~{len(entry_ids) / (2.3 * args.workers) / 60:.1f} min\n")
    else:
        pages = select_pages(pool, sample_pages, args.seed)
        cohort_source = None
        expected = len(pages) * PAGE_SIZE
        print(f"Gameweek {gw} | rank band top {pool:,} | {len(pages):,} pages -> ~{expected:,} managers")
        print(f"Estimated runtime ~{(len(pages) + expected) / (2.3 * args.workers) / 60:.1f} min\n")
        entry_ids = fetch_cohort(session, pages, args.workers)

    squads = fetch_many(session, [f"{API}/entry/{e}/event/{gw}/picks/" for e in entry_ids], args.workers, "picks")
    tallies, sampled, chips = aggregate(squads)
    if sampled == 0:
        message = "No squads retrieved - the gameweek may predate the cohort's entries."
        if strict:
            raise SystemExit(message)
        # Under --auto this must not abort the run: the remaining gameweeks in the window
        # are independent, and a transient API failure here would otherwise cost them too.
        print(f"GW{gw}{f' (cohort {from_cohort})' if label_pull else ''}: {message}")
        return False

    table = build_table(bootstrap, tallies, sampled)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    meta = {
        "gameweek": gw,
        "pool": pool,
        "pages_requested": len(pages),
        "entries_listed": len(entry_ids),
        "squads_sampled": sampled,
        "sampled_pages": sample_pages is not None,
        "seed": args.seed,
        "forced": args.force,
        "chips": dict(chips),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort_source": cohort_source,
        # Only a fresh pull freezes a cohort; a label pull points back at the one it reused.
        "entry_ids": [] if label_pull else entry_ids,
    }
    (OUT_DIR / f"cohort_{tag}.json").write_text(json.dumps(meta))

    print(f"\nSampled {sampled:,} squads ({len(entry_ids) - sampled:,} unavailable) in {time.time() - start:.0f}s")
    print(f"Chips: {', '.join(f'{k} {100 * v / sampled:.1f}%' for k, v in chips.most_common())}")
    if sample_pages is not None:
        print(f"Sampled cohort - 95% margin of error at most +/-{100 * 1.96 * math.sqrt(0.25 / sampled):.2f} pts (wider for clustered picks)")
    if args.top:
        print(f"\nTop {args.top} by EO, gameweek {gw}, top {pool:,}:\n")
        print(table.head(args.top).to_string(index=False))
    print(f"\nWritten to {out_path}")
    if label_pull:
        print(f"Label for GW{gw} measured against the GW{from_cohort} cohort")
    else:
        print(f"Cohort IDs frozen in {OUT_DIR / f'cohort_{tag}.json'}")
    return True


def run_auto(session: requests.Session, bootstrap: dict, args: argparse.Namespace) -> None:
    """Do whichever pulls are currently possible and not already captured.

    Each gameweek needs two pulls, gated differently: the label (previous cohort measured in
    this gameweek) becomes available once the deadline passes, the t=0 state once the
    gameweek finalises. Running this daily lands each one shortly after it becomes possible,
    and does nothing the rest of the time.
    """
    now = datetime.now(UTC)
    passed = [e for e in bootstrap["events"] if datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")) <= now]
    if not passed:
        print("No gameweek deadline has passed yet - nothing to do.")
        return

    recent = sorted(passed, key=lambda e: e["id"])[-args.auto_window :]
    print(f"Auto mode: checking GW{recent[0]['id']}-{recent[-1]['id']}\n")

    written = 0
    for event in recent:
        gw = event["id"]
        if gw > 1:
            written += run_pull(session, bootstrap, args, gw, gw - 1, strict=False)
        written += run_pull(session, bootstrap, args, gw, None, strict=False)
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
        run_pull(session, bootstrap, args, resolve_gameweek(bootstrap, args.gw), args.from_cohort)


if __name__ == "__main__":
    main()
