"""Log the FPL transfer market over time, so intra-week flow can be modelled later.

`transfers_in_event` / `transfers_out_event` are cumulative counters that reset at each
deadline, and the API exposes only their current value. Final totals are recoverable
afterwards from `element-summary`, but the intra-week trajectory is not - it exists only
if it was sampled while the window was open. That trajectory is the feature set an EO
prediction model needs, which is what this logger captures.

Snapshots are keyed by the gameweek pending transfers apply to (the next gameweek whose
deadline has not passed), so the log rolls over on its own once a deadline goes by.

Layout: each poll appends one small gzipped CSV under snapshots/gw{N}/. Separate small
files rather than one growing file keeps git history cheap - measured at ~1.5MB per
gameweek packed, against ~27MB for re-committing a growing consolidated file. Gzipped CSV
beats parquet at this size (7KB vs 21KB per poll; parquet's footer overhead dominates at
600 rows). Once the window closes, --compact merges the parts into a single parquet, which
is 4x smaller and ~15x faster to read than the equivalent CSV, and supports column pruning.

Because the counters are cumulative, a missed run costs resolution inside that gap but no
level - the next successful run reports the full running total. Polling faster than every
5 minutes is pointless: bootstrap-static is CDN-cached with max-age=300, so repeat calls
inside that window return byte-identical data. Unchanged payloads are skipped by default.

Modes:
  (default)      take one snapshot
  --watch        take one snapshot, then keep polling while inside the pre-deadline window
  --compact GW   merge that gameweek's parts into one parquet
  --health       verify the open window is being logged, exit non-zero if not
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from paths import DATA_DIR

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
SNAPSHOT_DIR = Path(os.environ.get("FPL_SNAPSHOT_DIR", DATA_DIR / "effective_ownership" / "snapshots"))
PART_SUFFIX = ".csv.gz"

# Per-player fields worth a time series. The price_change_* fields are inert as of
# 2026/27 (all zero, calibrating false) but are logged so their activation is captured.
PLAYER_FIELDS = [
    "id",
    "transfers_in_event",
    "transfers_out_event",
    "transfers_in",
    "transfers_out",
    "selected_by_percent",
    "now_cost",
    "cost_change_event",
    "price_change_percent",
    "price_change_hourly_rate",
    "price_change_calibrating",
    "status",
    "chance_of_playing_next_round",
    "ep_next",
    "form",
]


def fetch_bootstrap() -> dict:
    response = requests.get(BOOTSTRAP_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return response.json()


def deadline_of(event: dict) -> datetime:
    return datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))


def target_event(bootstrap: dict, now: datetime) -> dict:
    """The gameweek pending transfers apply to: the earliest whose deadline is still ahead."""
    upcoming = [e for e in bootstrap["events"] if deadline_of(e) > now]
    if not upcoming:
        raise SystemExit("No gameweek deadline remains - the season is over.")
    return min(upcoming, key=deadline_of)


def window_opened_at(bootstrap: dict, now: datetime) -> datetime | None:
    """When the current transfer window opened: the most recent deadline that has passed."""
    passed = [deadline_of(e) for e in bootstrap["events"] if deadline_of(e) <= now]
    return max(passed) if passed else None


def build_snapshot(bootstrap: dict, event: dict, now: datetime) -> pd.DataFrame:
    frame = pd.DataFrame([{f: element.get(f) for f in PLAYER_FIELDS} for element in bootstrap["elements"]])
    frame.insert(0, "secs_to_deadline", int((deadline_of(event) - now).total_seconds()))
    frame.insert(0, "event", event["id"])
    frame.insert(0, "captured_at", now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    return frame


def read_marker(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def take_snapshot(args: argparse.Namespace) -> tuple[int, bool]:
    """One poll. Returns (seconds to deadline, whether a part was written)."""
    now = datetime.now(UTC)
    bootstrap = fetch_bootstrap()
    event = target_event(bootstrap, now)
    frame = build_snapshot(bootstrap, event, now)

    marker_path = SNAPSHOT_DIR / f"gw{event['id']}.latest.json"
    previous = read_marker(marker_path)
    total_in = int(frame["transfers_in_event"].sum())
    total_out = int(frame["transfers_out_event"].sum())
    secs_left = int(frame["secs_to_deadline"].iloc[0])

    delta = f"+{total_in - previous['total_in']:,} since {previous.get('captured_at', '?')}" if "total_in" in previous else "no prior snapshot"
    print(f"{now:%Y-%m-%d %H:%M:%SZ} | GW{event['id']} | deadline in {secs_left / 3600:.1f}h | {total_in:,} in ({delta})")

    if args.top:
        names = {e["id"]: e["web_name"] for e in bootstrap["elements"]}
        movers = frame.assign(net=frame["transfers_in_event"] - frame["transfers_out_event"]).nlargest(args.top, "net")
        for _, row in movers.iterrows():
            print(f"  {names.get(row['id'], row['id']):<16} net {int(row['net']):>+9,}  ({row['selected_by_percent']}% owned)")

    if args.dry_run:
        print("  dry run - nothing written")
        return secs_left, False

    if previous.get("total_in") == total_in and previous.get("total_out") == total_out and not args.force:
        print("  payload identical to last snapshot (CDN cache) - skipped")
        return secs_left, False

    window_dir = SNAPSHOT_DIR / f"gw{event['id']}"
    window_dir.mkdir(parents=True, exist_ok=True)
    out_path = window_dir / f"market_{now.strftime('%Y%m%dT%H%M%S%f')[:-3]}Z{PART_SUFFIX}"
    frame.to_csv(out_path, index=False, compression="gzip")
    marker_path.write_text(json.dumps({"captured_at": frame["captured_at"].iloc[0], "event": event["id"], "total_in": total_in, "total_out": total_out}))
    print(f"  wrote {out_path.name} ({out_path.stat().st_size / 1024:.0f}KB, {len(list(window_dir.glob('*' + PART_SUFFIX)))} this window)")
    return secs_left, True


def run_watch(args: argparse.Namespace) -> None:
    """Poll once, then keep polling at a tighter interval while inside the pre-deadline window.

    Cron alone cannot track the deadline, which moves between gameweeks (GW4 is Saturday
    12:30Z, not the usual 17:30Z). Once this job is running it polls on its own clock, so
    the densest sampling lands where the flow curve is steepest regardless of cron drift.
    """
    stop_at = time.monotonic() + args.max_runtime * 60
    while True:
        secs_left, _ = take_snapshot(args)
        if not args.watch:
            return
        if secs_left <= 0:
            print("Deadline passed - stopping.")
            return
        if secs_left > args.dense_window * 3600:
            print(f"Outside the {args.dense_window}h dense window - stopping until the next scheduled run.")
            return
        if time.monotonic() + args.dense_interval * 60 > stop_at:
            print(f"Reached the {args.max_runtime} min runtime cap - stopping until the next scheduled run.")
            return
        print(f"  ...inside the dense window, next poll in {args.dense_interval} min")
        time.sleep(args.dense_interval * 60)


def run_compact(args: argparse.Namespace) -> None:
    """Merge one window's per-poll parts into a single parquet for training."""
    window_dir = SNAPSHOT_DIR / f"gw{args.compact}"
    parts = sorted(window_dir.glob("*" + PART_SUFFIX))
    if not parts:
        raise SystemExit(f"No snapshots found in {window_dir}")

    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    frame = frame.drop_duplicates(["captured_at", "id"]).sort_values(["captured_at", "id"]).reset_index(drop=True)

    out_path = SNAPSHOT_DIR / f"market_gw{args.compact}.parquet"
    frame.to_parquet(out_path, index=False, compression="zstd")
    print(f"Merged {len(parts)} parts ({frame['captured_at'].nunique()} distinct snapshots) -> {out_path.name}")
    print(f"  {len(frame):,} rows, {out_path.stat().st_size / 1024:.0f}KB (parts were {sum(p.stat().st_size for p in parts) / 1024:.0f}KB)")

    if args.prune:
        for part in parts:
            part.unlink()
        window_dir.rmdir()
        print(f"  pruned {len(parts)} part files")


def first_snapshot_time(parts: list[Path]) -> datetime | None:
    """Timestamp of the earliest part, parsed from its filename."""
    if not parts:
        return None
    stamp = parts[0].name.removeprefix("market_").removesuffix(PART_SUFFIX).rstrip("Z")
    return datetime.strptime(stamp, "%Y%m%dT%H%M%S%f").replace(tzinfo=UTC)


def run_health(args: argparse.Namespace) -> None:
    """Fail loudly if the open window is not being logged.

    A workflow that errors sends a notification, but a skipped schedule or a job that runs
    and writes nothing is silent - and lost trajectory is unrecoverable. This turns that
    silence into a failure.
    """
    now = datetime.now(UTC)
    bootstrap = fetch_bootstrap()
    event = target_event(bootstrap, now)
    opened = window_opened_at(bootstrap, now)

    parts = sorted((SNAPSHOT_DIR / f"gw{event['id']}").glob("*" + PART_SUFFIX))
    hours_open = (now - opened).total_seconds() / 3600 if opened else 0.0
    newest = read_marker(SNAPSHOT_DIR / f"gw{event['id']}.latest.json").get("captured_at")

    # Coverage is measured from the first snapshot of the window, not from when the window
    # opened - otherwise activating the logger mid-week always reports a false failure.
    logging_since = first_snapshot_time(parts)
    hours_logging = (now - logging_since).total_seconds() / 3600 if logging_since else 0.0
    expected = max(1, int(hours_logging))

    print(f"GW{event['id']} window open {hours_open:.1f}h | logging for {hours_logging:.1f}h | {len(parts)} snapshots | expected ~{expected} | newest {newest or 'none'}")

    problems = []
    if len(parts) < expected * args.min_coverage:
        problems.append(f"only {len(parts)} snapshots across {hours_logging:.1f}h of logging (expected at least {expected * args.min_coverage:.0f})")
    if newest:
        age = (now - datetime.fromisoformat(newest.replace("Z", "+00:00"))).total_seconds() / 3600
        if age > args.max_age:
            problems.append(f"newest snapshot is {age:.1f}h old (limit {args.max_age}h)")
    elif hours_open > args.max_age:
        problems.append("no snapshots at all for the open window")

    if problems:
        raise SystemExit("SNAPSHOT LOGGER UNHEALTHY:\n  - " + "\n  - ".join(problems))
    print("healthy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log, compact, or health-check FPL transfer-market snapshots.")
    parser.add_argument("--watch", action="store_true", help="keep polling while inside the pre-deadline dense window")
    parser.add_argument("--dense-window", type=float, default=6.0, help="hours before the deadline to poll densely (default: 6)")
    parser.add_argument("--dense-interval", type=float, default=10.0, help="minutes between dense polls (default: 10)")
    parser.add_argument("--max-runtime", type=float, default=50.0, help="minutes before yielding to the next scheduled run (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="fetch and summarise without writing")
    parser.add_argument("--force", action="store_true", help="log even if the payload is unchanged since the last snapshot")
    parser.add_argument("--top", type=int, default=5, help="movers to print, 0 to suppress (default: 5)")
    parser.add_argument("--compact", type=int, metavar="GW", default=None, help="merge that gameweek's parts into one parquet instead of polling")
    parser.add_argument("--prune", action="store_true", help="with --compact, delete the parts afterwards")
    parser.add_argument("--health", action="store_true", help="check the open window is being logged, exit non-zero if not")
    parser.add_argument("--min-coverage", type=float, default=0.5, help="with --health, minimum fraction of expected snapshots (default: 0.5)")
    parser.add_argument("--max-age", type=float, default=3.0, help="with --health, maximum hours since the newest snapshot (default: 3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compact is not None:
        run_compact(args)
    elif args.health:
        run_health(args)
    else:
        run_watch(args)


if __name__ == "__main__":
    main()
