"""Collect one raw FPL bootstrap-static snapshot into S3.

Zero third-party dependencies: urllib and gzip are stdlib, boto3 ships with the Lambda runtime.
Deployment is therefore a plain zip - no build step, no layers, no vendored wheels.

The function deliberately does as little as possible. Every transformation applied here is a
decision that cannot be revisited once the moment has passed, so what gets stored is the
unmodified payload; all extraction happens later, locally, where it can be redone against data
that is already safe.

Failures are not swallowed. A raised exception is what drives the retry policy, the dead-letter
queue and the error alarm, so a transient API problem becomes visible rather than silently
producing a gap.
"""

import gzip
import json
import os
import urllib.request
from datetime import UTC, datetime

DEFAULT_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# A CDN error page is still HTTP 200 with a body. Requiring these keys means such a response
# fails loudly instead of being stored as though it were a snapshot.
REQUIRED_KEYS = ("elements", "events", "teams")

_s3 = None


def s3_client():
    """Created lazily, and boto3 imported lazily with it.

    boto3 is supplied by the Lambda runtime, never packaged. Importing it here rather than at
    module scope means the module has no import-time third-party dependency at all, so the
    tests run anywhere without installing the AWS SDK.
    """
    global _s3
    if _s3 is None:
        import boto3

        _s3 = boto3.client("s3")
    return _s3


def fetch(url: str, user_agent: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return response.read(), dict(response.headers)


def validate(body: bytes) -> dict:
    """Parse and sanity-check the payload. Raises rather than returning a failure flag."""
    payload = json.loads(body)
    missing = [key for key in REQUIRED_KEYS if not payload.get(key)]
    if missing:
        raise ValueError(f"payload missing required key(s): {', '.join(missing)}")
    return payload


def pending_event(payload: dict, now: datetime) -> int | None:
    """The gameweek transfers currently apply to: the earliest deadline still ahead."""
    upcoming = [
        event
        for event in payload["events"]
        if datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00")) > now
    ]
    if not upcoming:
        return None
    return min(upcoming, key=lambda e: e["deadline_time"])["id"]


def key_for(prefix: str, now: datetime, cdn_age: int | None = None) -> str:
    """Microsecond resolution, not seconds: the retry policy can fire a second attempt inside
    the same second, and a colliding key would silently overwrite the earlier snapshot rather
    than failing. Keys remain lexicographically ordered by time.

    The CDN Age is appended because `aws s3 sync` does not carry object metadata down to local
    files - anything the local build needs has to live in the key or the body. Origin time is
    captured_at minus this, which is what makes two samples ten minutes apart distinguishable
    from two reads of the same cached copy.
    """
    stem = f"{prefix}/{now:%Y/%m/%d/%H%M%S%f}"
    return f"{stem}-a{cdn_age}.json.gz" if cdn_age is not None else f"{stem}.json.gz"


def build_metadata(payload: dict, headers: dict, now: datetime) -> dict[str, str]:
    """Small, ASCII-only object metadata. Everything here is also derivable from the body;
    it exists so the bucket can be surveyed without downloading and decompressing objects.

    cdn_age is the CDN's Age header. bootstrap-static is cached with max-age=300, so two
    objects ten minutes apart are not necessarily ten minutes apart at the source - this
    records how stale each copy was, which matters when reconstructing the flow trajectory.
    """
    metadata = {
        "captured-at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elements": str(len(payload.get("elements", []))),
    }
    event_id = pending_event(payload, now)
    if event_id is not None:
        metadata["pending-event"] = str(event_id)
    for header, name in (("Age", "cdn-age"), ("ETag", "cdn-etag")):
        value = headers.get(header)
        if value:
            metadata[name] = str(value).strip('"')[:128]
    return metadata


def handler(event, context):
    bucket = os.environ["BUCKET"]
    prefix = os.environ.get("PREFIX", "raw")
    url = os.environ.get("API_URL", DEFAULT_API_URL)
    user_agent = os.environ.get("USER_AGENT", "Mozilla/5.0")
    timeout = float(os.environ.get("TIMEOUT_SECONDS", "20"))

    now = datetime.now(UTC)
    body, headers = fetch(url, user_agent, timeout)
    payload = validate(body)
    age = headers.get("Age")
    key = key_for(prefix, now, int(age) if age and str(age).isdigit() else None)

    # Unconditional write. Skipping unchanged payloads would save a little storage at the cost
    # of being unable to distinguish "ran and skipped" from "never ran" - and the second is
    # exactly the failure this whole system exists to detect.
    s3_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=gzip.compress(body),
        ContentType="application/json",
        ContentEncoding="gzip",
        Metadata=build_metadata(payload, headers, now),
    )

    result = {"key": key, "bytes": len(body), "elements": len(payload["elements"])}
    print(json.dumps({"ok": True, **result}))
    return result
