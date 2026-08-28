"""Unit tests for the collector Lambda.

The handler is small on purpose, so these cover the parts that can actually go wrong: a
response that is HTTP 200 but not a snapshot, the key convention, and the fact that the write
is unconditional.
"""

import gzip
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "collector"))
import app  # noqa: E402


def payload(events=None):
    return {
        "elements": [{"id": 1, "web_name": "Test"}],
        "teams": [{"id": 1, "short_name": "TST"}],
        "events": events
        or [
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"},
            {"id": 2, "deadline_time": "2026-08-28T17:30:00Z"},
            {"id": 3, "deadline_time": "2026-09-04T17:30:00Z"},
        ],
    }


def test_validate_rejects_a_cdn_error_page():
    # HTTP 200 with an HTML body is the dangerous case: without this it would be stored as
    # though it were a snapshot.
    with pytest.raises(json.JSONDecodeError):
        app.validate(b"<html><body>503 Service Unavailable</body></html>")


@pytest.mark.parametrize("missing", ["elements", "events", "teams"])
def test_validate_rejects_payload_missing_a_required_key(missing):
    body = payload()
    del body[missing]
    with pytest.raises(ValueError, match=missing):
        app.validate(json.dumps(body).encode())


def test_validate_rejects_empty_elements():
    body = payload()
    body["elements"] = []
    with pytest.raises(ValueError, match="elements"):
        app.validate(json.dumps(body).encode())


def test_key_is_timestamp_ordered():
    early = app.key_for("raw", datetime(2026, 8, 28, 4, 15, 3, tzinfo=UTC))
    late = app.key_for("raw", datetime(2026, 8, 28, 17, 30, 0, tzinfo=UTC))
    assert early == "raw/2026/08/28/041503000000.json.gz"
    assert app.key_for("raw", early_dt := datetime(2026, 8, 28, 4, 15, 3, tzinfo=UTC), 284) == "raw/2026/08/28/041503000000-a284.json.gz"
    assert early < late  # lexicographic order matches chronological order


def test_keys_within_the_same_second_do_not_collide():
    # The retry policy can fire a second attempt inside the same second. At second resolution
    # the retry would overwrite the first snapshot silently.
    base = datetime(2026, 8, 28, 4, 15, 3, tzinfo=UTC)
    assert app.key_for("raw", base) != app.key_for("raw", base.replace(microsecond=250000))


def test_pending_event_is_the_next_deadline_not_the_last():
    now = datetime(2026, 8, 28, 4, 0, tzinfo=UTC)
    assert app.pending_event(payload(), now) == 2


def test_pending_event_is_none_after_the_final_deadline():
    now = datetime(2027, 6, 1, tzinfo=UTC)
    assert app.pending_event(payload(), now) is None


def test_metadata_is_ascii_strings_only():
    now = datetime(2026, 8, 28, 4, 15, tzinfo=UTC)
    meta = app.build_metadata(payload(), {"Age": "42", "ETag": '"abc123"'}, now)
    assert meta["captured-at"] == "2026-08-28T04:15:00Z"
    assert meta["pending-event"] == "2"
    assert meta["cdn-age"] == "42"
    assert meta["cdn-etag"] == "abc123"
    for key, value in meta.items():
        assert isinstance(value, str) and value.isascii(), (key, value)


def test_handler_writes_gzipped_body_unconditionally(monkeypatch):
    body = json.dumps(payload()).encode()
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setattr(app, "fetch", lambda *a, **k: (body, {"Age": "7"}))

    written = []

    class FakeS3:
        def put_object(self, **kwargs):
            written.append(kwargs)

    monkeypatch.setattr(app, "s3_client", lambda: FakeS3())

    first = app.handler({}, None)
    second = app.handler({}, None)  # identical payload must still be written

    assert len(written) == 2, "unchanged payloads must not be skipped"
    assert gzip.decompress(written[0]["Body"]) == body
    assert written[0]["ContentEncoding"] == "gzip"
    assert written[0]["Key"].startswith("raw/")
    assert first["elements"] == 1 and second["elements"] == 1


def test_handler_propagates_fetch_failure(monkeypatch):
    # Failures must raise so the retry policy, DLQ and error alarm all engage.
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setattr(app, "fetch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        app.handler({}, None)
