"""Build one training-ready market table from raw S3 snapshots and legacy GitHub parts.

Two collectors produced this dataset. Up to the GW3 deadline a GitHub Actions job wrote
extracted CSV parts to the `data` branch; after it, an AWS Lambda writes the raw
bootstrap-static payload to S3. This reads both and emits a single table, so the seam between
them does not reach the model.

Conversion goes forward only - legacy CSV is projected into the shared schema, never
back-converted into synthetic raw JSON. The legacy parts carry only PLAYER_FIELDS, so any
column derived from a field outside that list is simply absent for the early gameweeks; that
is a real limitation of the old collector and is surfaced rather than papered over.

Both sources now live in S3: raw/ holds the Lambda payloads, legacy/ holds the compacted
GitHub windows migrated off the `data` branch. One sync gets everything.

    aws s3 sync s3://<bucket> ./data/market
    python run/build_market_table.py --raw data/market/raw \\
        --legacy data/market/legacy -o data/market_table.parquet
"""

import argparse
import gzip
import json
from datetime import UTC, datetime, timedelta
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
# obs_time is derived after both sources are concatenated, so it is not in LEAD_COLUMNS
# (which is the set the legacy reader must find in a CSV part).
OUTPUT_COLUMNS = ["captured_at", "obs_time", "event", "secs_to_deadline", "source", "cdn_age"] + PLAYER_FIELDS

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
    # A cached payload describes the market as it stood cdn_age seconds ago, so the window it
    # belongs to follows from that instant, not from the fetch. Using the fetch time labels a
    # pre-deadline payload with the next gameweek whenever a stale copy straddles a deadline.
    observed = now - timedelta(seconds=cdn_age) if cdn_age is not None else now
    event = pending_event(payload.get("events", []), observed)
    if event is None:
        return None  # season over; nothing pending to attribute the transfers to

    frame = pd.DataFrame(elements)
    missing = [c for c in PLAYER_FIELDS if c not in frame.columns]
    for column in missing:
        frame[column] = pd.NA
    frame = frame[PLAYER_FIELDS].copy()
    frame.insert(0, "cdn_age", cdn_age if cdn_age is not None else pd.NA)
    frame.insert(0, "source", "s3")
    frame.insert(0, "secs_to_deadline", int((deadline_of(event) - observed).total_seconds()))
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


def add_obs_time(table: pd.DataFrame) -> pd.DataFrame:
    """When the numbers were true, as distinct from when we fetched them.

    bootstrap-static is CDN-cached, so a fetch often returns a copy minted earlier - median
    157s over the GW2-GW3 overlap, and up to 2252s. Two fetches five minutes apart can carry
    payloads one second apart, and ordering on captured_at misorders 130 of 874 consecutive
    pairs outright. Any feature built as a difference between neighbouring rows (transfer
    rate, price momentum, the deadline ramp) is computed backwards at those points, so
    obs_time rather than captured_at is the column to sort and diff on.

    The legacy collector never read the Age header, so its rows fall back to captured_at;
    their null cdn_age is the signal that the timing is approximate to within ~40 minutes.
    """
    captured = pd.to_datetime(table["captured_at"], format="ISO8601", utc=True)
    lag = pd.to_timedelta(table["cdn_age"].fillna(0).astype("int64"), unit="s")
    stamp = (captured - lag).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    table["obs_time"] = stamp.astype("string")
    return table


def drop_stale_rollover(table: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """Drop snapshots that still carry the previous window's counters.

    `event` flips at the deadline but transfers_*_event does not reset at the same instant.
    At the GW2->GW3 boundary the API served GW2's closing total (10,727,201) under event 3
    for at least fifteen minutes before resetting to 30,524. Both collectors recorded this
    identically, so it is API behaviour, not a pipeline fault - which is why it is corrected
    here rather than in either collector.

    Left in, a window's net-transfer feature reads as a phantom nine-million outflow at
    exactly the boundary a model is trying to predict across, and it recurs every deadline.

    Within one event the summed counter is non-decreasing, so a leading snapshot whose total
    exceeds anything that follows cannot belong to this window. Only leading snapshots are
    considered: a mid-window anomaly is a different problem and is left visible.
    """
    totals = (
        table.groupby(["event", "captured_at"], dropna=False)
        .agg(total=("transfers_in_event", "sum"), obs_time=("obs_time", "first"))
        .reset_index()
        .sort_values(["event", "obs_time"], kind="mergesort")
    )
    stale: list[tuple] = []
    for event, group in totals.groupby("event", dropna=False, sort=True):
        values = group["total"].astype("float64").to_numpy()
        fetched = group["captured_at"].to_numpy()
        observed = group["obs_time"].to_numpy()
        for i in range(len(values) - 1):
            if values[i] > values[i + 1 :].min():
                stale.append((event, fetched[i], observed[i], values[i]))
            else:
                break  # the window has started counting; anything later is this window's
    if not stale:
        return table, []
    keys = {(event, fetch) for event, fetch, _, _ in stale}
    mask = pd.Series(list(zip(table["event"], table["captured_at"])), index=table.index).isin(keys)
    return table.loc[~mask].copy(), stale


def collect(raw_dir: Path | None, legacy_dir: Path | None, keep_rollover: bool = False) -> pd.DataFrame:
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

    table = add_obs_time(table)
    # A cached payload fetched twice is one observation, not two. Those rows differ in
    # captured_at so the dedupe above cannot see them, but they share an obs_time to the
    # millisecond - and left in they double-count that instant in any per-snapshot aggregate.
    before = table["obs_time"].nunique(), len(table)
    table = (
        table.sort_values(["obs_time", "id", "captured_at"], kind="mergesort")
        .drop_duplicates(["obs_time", "id"], keep="first")
        .reset_index(drop=True)
    )
    if len(table) != before[1]:
        print(f"({before[1] - len(table):,} row(s) dropped: same cached payload fetched more than once)")

    if not keep_rollover:
        table, stale = drop_stale_rollover(table)
        for event, fetch, observed, total in stale:
            print(f"  dropped {fetch} (obs {observed}, event {event}): counters still at {total:,.0f}, not yet reset")
        if stale:
            print(f"({len(stale)} stale-rollover snapshot(s) dropped; --keep-rollover to retain)")

    return table[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=None, help="directory of S3 raw/*.json.gz objects")
    parser.add_argument("--legacy", type=Path, default=None, help="data-branch snapshots/ directory")
    parser.add_argument("-o", "--out", type=Path, default=Path("data/market_table.parquet"))
    parser.add_argument("--keep-rollover", action="store_true",
                        help="keep post-deadline snapshots whose counters had not reset yet")
    args = parser.parse_args()

    table = collect(args.raw, args.legacy, keep_rollover=args.keep_rollover)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False, compression="zstd")

    snapshots = table["captured_at"].nunique()
    print(f"\n{len(table):,} rows | {snapshots:,} snapshots | events {sorted(table['event'].unique())}")
    print(table.groupby("source")["captured_at"].agg(["nunique", "min", "max"]).to_string())
    print(f"\nwritten to {args.out} ({args.out.stat().st_size / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
