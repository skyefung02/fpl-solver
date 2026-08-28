"""Build one training-ready market table from raw S3 snapshots and legacy GitHub parts.

Two collectors produced this dataset. Up to the GW3 deadline a GitHub Actions job wrote
extracted CSV parts to the `data` branch; after it, an AWS Lambda writes the raw
bootstrap-static payload to S3. This reads both and emits a single table, so the seam between
them does not reach the model.

Conversion goes forward only - legacy CSV is projected into the shared schema, never
back-converted into synthetic raw JSON. The legacy parts carry only PLAYER_FIELDS, so any
column derived from a field outside that list is simply absent for the early gameweeks; that
is a real limitation of the old collector and is surfaced rather than papered over.

Typical use:

    aws s3 sync s3://<bucket>/raw ./data/market_raw
    python run/build_market_table.py --raw data/market_raw \\
        --legacy /path/to/data-branch/snapshots -o data/market_table.parquet
"""

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# The columns the legacy collector captured. The raw payloads contain far more; this is the
# intersection, and therefore the schema both sources can honestly fill.
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
LEAD_COLUMNS = ["captured_at", "event", "secs_to_deadline", "source", "cdn_age"]

# The API returns several numeric quantities as JSON strings ("38.3"), while pandas infers them
# as floats when reading the legacy CSV. Left alone the two sources concat to object dtype and
# parquet rejects the frame, so both readers normalise here rather than trusting inference.
STRING_FIELDS = {"status"}
BOOL_FIELDS = {"price_change_calibrating"}


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Give both sources identical dtypes, so the seam between them is invisible downstream."""
    for column in PLAYER_FIELDS:
        if column in STRING_FIELDS:
            frame[column] = frame[column].astype("string")
        elif column in BOOL_FIELDS:
            frame[column] = frame[column].map({True: True, False: False, "True": True, "False": False}).astype("boolean")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["event"] = pd.to_numeric(frame["event"], errors="coerce").astype("Int64")
    frame["secs_to_deadline"] = pd.to_numeric(frame["secs_to_deadline"], errors="coerce").astype("Int64")
    frame["cdn_age"] = pd.to_numeric(frame["cdn_age"], errors="coerce").astype("Int64")
    frame["captured_at"] = frame["captured_at"].astype("string")
    frame["source"] = frame["source"].astype("string")
    return frame


def deadline_of(event: dict) -> datetime:
    return datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))


def pending_event(events: list[dict], now: datetime) -> dict | None:
    upcoming = [e for e in events if deadline_of(e) > now]
    return min(upcoming, key=deadline_of) if upcoming else None


def captured_at_from_key(path: Path) -> tuple[datetime, int | None]:
    """raw/YYYY/MM/DD/HHMMSSffffff-aNNN.json.gz -> (capture instant, CDN age or None)."""
    stamp, _, age = path.name.split(".")[0].partition("-a")
    day = path.parent
    when = datetime.strptime(
        f"{day.parent.parent.name}{day.parent.name}{day.name}{stamp}", "%Y%m%d%H%M%S%f"
    ).replace(tzinfo=UTC)
    return when, (int(age) if age.isdigit() else None)


def read_raw(path: Path) -> pd.DataFrame | None:
    """One raw bootstrap-static payload -> rows in the shared schema."""
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    elements = payload.get("elements")
    if not elements:
        return None
    now, cdn_age = captured_at_from_key(path)
    event = pending_event(payload.get("events", []), now)
    if event is None:
        return None  # season over; nothing pending to attribute the transfers to

    frame = pd.DataFrame(elements)
    missing = [c for c in PLAYER_FIELDS if c not in frame.columns]
    for column in missing:
        frame[column] = pd.NA
    frame = frame[PLAYER_FIELDS].copy()
    frame.insert(0, "cdn_age", cdn_age if cdn_age is not None else pd.NA)
    frame.insert(0, "source", "s3")
    frame.insert(0, "secs_to_deadline", int((deadline_of(event) - now).total_seconds()))
    frame.insert(0, "event", event["id"])
    frame.insert(0, "captured_at", now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    return normalise(frame)


def read_legacy(path: Path) -> pd.DataFrame | None:
    """One legacy CSV part or compacted parquet -> rows in the shared schema."""
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if frame.empty:
        return None
    for column in PLAYER_FIELDS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["source"] = "github"
    frame["cdn_age"] = pd.NA
    return normalise(frame[LEAD_COLUMNS + PLAYER_FIELDS].copy())


def collect(raw_dir: Path | None, legacy_dir: Path | None) -> pd.DataFrame:
    frames, skipped = [], 0

    if raw_dir:
        files = sorted(raw_dir.rglob("*.json.gz"))
        print(f"raw:    {len(files):,} objects under {raw_dir}")
        for path in files:
            try:
                frame = read_raw(path)
            except (OSError, ValueError, KeyError) as error:
                print(f"  skipped {path.name}: {error}")
                skipped += 1
                continue
            if frame is not None:
                frames.append(frame)

    if legacy_dir:
        files = sorted(legacy_dir.rglob("*.csv.gz")) + sorted(legacy_dir.glob("*.parquet"))
        print(f"legacy: {len(files):,} parts under {legacy_dir}")
        for path in files:
            try:
                frame = read_legacy(path)
            except (OSError, ValueError, KeyError) as error:
                print(f"  skipped {path.name}: {error}")
                skipped += 1
                continue
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise SystemExit("No input found. Pass --raw and/or --legacy.")
    if skipped:
        print(f"({skipped} unreadable file(s) skipped)")

    table = pd.concat(frames, ignore_index=True)
    # The two collectors overlap during the parallel-run window. Identical (captured_at, id)
    # rows are the same observation seen twice; prefer the raw source, which carries the whole
    # payload and can be re-derived if the projection above ever changes.
    table["_rank"] = (table["source"] == "github").astype(int)
    table = (
        table.sort_values(["captured_at", "id", "_rank"])
        .drop_duplicates(["captured_at", "id"], keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=None, help="directory of S3 raw/*.json.gz objects")
    parser.add_argument("--legacy", type=Path, default=None, help="data-branch snapshots/ directory")
    parser.add_argument("-o", "--out", type=Path, default=Path("data/market_table.parquet"))
    args = parser.parse_args()

    table = collect(args.raw, args.legacy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False, compression="zstd")

    snapshots = table["captured_at"].nunique()
    print(f"\n{len(table):,} rows | {snapshots:,} snapshots | events {sorted(table['event'].unique())}")
    print(table.groupby("source")["captured_at"].agg(["nunique", "min", "max"]).to_string())
    print(f"\nwritten to {args.out} ({args.out.stat().st_size / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
