"""Log the FPL transfer market over time, so intra-week flow can be modelled later.

`transfers_in_event` / `transfers_out_event` are cumulative counters that reset at each
deadline, and the API exposes only their current value. Final totals are recoverable
afterwards from `element-summary`, but the intra-week trajectory is not - it exists only
if it was sampled while the window was open. That trajectory is the feature set an EO
prediction model needs, which is what this logger captures.

Snapshots are keyed by the gameweek pending transfers apply to (the next gameweek whose
deadline has not passed), so the log rolls over on its own once a deadline goes by.

Layout: each poll writes one small parquet file under snapshots/gw{N}/. Small separate
files keep the git history cheap - an append-only stream of ~21KB blobs rather than a
growing file re-stored in full on every commit. Once the window closes, --compact merges
them into a single parquet, which is both 4x smaller and ~15x faster to read than the
equivalent gzipped CSV, and supports column pruning when training.

Because the counters are cumulative, a missed run costs resolution inside that gap but no
level - the next successful run reports the full running total. Polling faster than every
5 minutes is pointless: bootstrap-static is CDN-cached with max-age=300, so repeat calls
inside that window return byte-identical data. Unchanged payloads are skipped by default
to avoid logging the same observation twice.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from paths import DATA_DIR

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
SNAPSHOT_DIR = DATA_DIR / "effective_ownership" / "snapshots"
COMPRESSION = "zstd"

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


def target_event(bootstrap: dict, now: datetime) -> dict:
    """The gameweek pending transfers apply to: the earliest whose deadline is still ahead."""
    upcoming = [e for e in bootstrap["events"] if datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00")) > now]
    if not upcoming:
        raise SystemExit("No gameweek deadline remains - the season is over.")
    return min(upcoming, key=lambda e: e["deadline_time"])


def build_snapshot(bootstrap: dict, event: dict, now: datetime) -> pd.DataFrame:
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    frame = pd.DataFrame([{f: element.get(f) for f in PLAYER_FIELDS} for element in bootstrap["elements"]])
    frame.insert(0, "secs_to_deadline", int((deadline - now).total_seconds()))
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


def totals(frame: pd.DataFrame) -> tuple[int, int]:
    return int(frame["transfers_in_event"].sum()), int(frame["transfers_out_event"].sum())


def describe(frame: pd.DataFrame, names: dict[int, str], top: int) -> None:
    movers = frame.assign(net=frame["transfers_in_event"] - frame["transfers_out_event"]).nlargest(top, "net")
    for _, row in movers.iterrows():
        print(f"  {names.get(row['id'], row['id']):<16} net {int(row['net']):>+9,}  ({row['selected_by_percent']}% owned)")


def run_snapshot(args: argparse.Namespace) -> None:
    now = datetime.now(UTC)
    bootstrap = fetch_bootstrap()
    event = target_event(bootstrap, now)
    frame = build_snapshot(bootstrap, event, now)

    window_dir = SNAPSHOT_DIR / f"gw{event['id']}"
    marker_path = SNAPSHOT_DIR / f"gw{event['id']}.latest.json"
    previous = read_marker(marker_path)
    total_in, total_out = totals(frame)

    hours_left = frame["secs_to_deadline"].iloc[0] / 3600
    print(f"{now:%Y-%m-%d %H:%M:%SZ} | GW{event['id']} | deadline in {hours_left:.1f}h")

    if previous:
        delta = total_in - previous.get("total_in", 0)
        print(f"{total_in:,} transfers in this window, +{delta:,} since {previous.get('captured_at', '?')}")
    else:
        print(f"{total_in:,} transfers in this window (no prior snapshot to compare)")

    describe(frame, {e["id"]: e["web_name"] for e in bootstrap["elements"]}, args.top)

    if args.dry_run:
        print("\ndry run - nothing written")
        return

    unchanged = previous.get("total_in") == total_in and previous.get("total_out") == total_out
    if unchanged and not args.force:
        print("\nPayload identical to the last snapshot (CDN cache) - skipped. Use --force to log anyway.")
        return

    window_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%S%f")[:-3]
    out_path = window_dir / f"market_{stamp}Z.parquet"
    frame.to_parquet(out_path, index=False, compression=COMPRESSION)
    marker_path.write_text(json.dumps({"captured_at": frame["captured_at"].iloc[0], "event": event["id"], "total_in": total_in, "total_out": total_out}))

    parts = len(list(window_dir.glob("*.parquet")))
    print(f"\nWrote {out_path.relative_to(DATA_DIR)} ({len(frame)} rows, {out_path.stat().st_size / 1024:.0f}KB)")
    print(f"{parts} snapshot{'' if parts == 1 else 's'} logged for GW{event['id']} so far")


def run_compact(args: argparse.Namespace) -> None:
    """Merge one window's per-poll files into a single parquet for training."""
    window_dir = SNAPSHOT_DIR / f"gw{args.compact}"
    parts = sorted(window_dir.glob("*.parquet"))
    if not parts:
        raise SystemExit(f"No snapshots found in {window_dir}")

    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    frame = frame.drop_duplicates(["captured_at", "id"]).sort_values(["captured_at", "id"]).reset_index(drop=True)

    out_path = SNAPSHOT_DIR / f"market_gw{args.compact}.parquet"
    frame.to_parquet(out_path, index=False, compression=COMPRESSION)
    print(f"Merged {len(parts)} snapshots ({frame['captured_at'].nunique()} distinct) -> {out_path.name}")
    print(f"  {len(frame):,} rows, {out_path.stat().st_size / 1024:.0f}KB (parts were {sum(p.stat().st_size for p in parts) / 1024:.0f}KB)")

    if args.prune:
        for part in parts:
            part.unlink()
        window_dir.rmdir()
        print(f"  pruned {len(parts)} part files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log or compact FPL transfer-market snapshots.")
    parser.add_argument("--dry-run", action="store_true", help="fetch and summarise without writing")
    parser.add_argument("--force", action="store_true", help="log even if the payload is unchanged since the last snapshot")
    parser.add_argument("--top", type=int, default=5, help="movers to print (default: 5)")
    parser.add_argument("--compact", type=int, metavar="GW", default=None, help="merge that gameweek's snapshots into one parquet instead of polling")
    parser.add_argument("--prune", action="store_true", help="with --compact, delete the per-poll files afterwards")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compact is not None:
        run_compact(args)
    else:
        run_snapshot(args)


if __name__ == "__main__":
    main()
