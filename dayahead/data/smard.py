"""
SMARD API client and ingestion.

The raw store is the source of truth. Responses are written to disk exactly
as returned, including the meta_data block, so parsing decisions stay
reversible and provenance is preserved.

Ingestion is cached, resumable and idempotent. A block already on disk is
never refetched, an interrupted run continues where it stopped, and running
twice gives the same result as running once.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from .. import config as cfg

TIMEOUT = 60
RETRIES = 4
BACKOFF = 2.0
POLITE_DELAY = 0.15
USER_AGENT = "dayahead-ingest/0.1"

# Blocks at the end of the index are always refetched, never served from
# cache.
#
# SMARD publishes weekly blocks and keeps filling the current one as the week
# progresses, so a block downloaded on Monday holds Monday's data and stays
# that way on disk. The original cache rule skipped anything already stored,
# which made ingestion a no-op from the second run of any given week onward:
# the store looked complete, the summary reported every block present, and no
# new hour ever arrived. It was invisible until the daily forecast job asked
# for tomorrow and found nothing.
#
# Two blocks rather than one, because a run near a week boundary can find the
# previous block still incomplete.
ALWAYS_REFETCH_BLOCKS = 2


def get_json(url: str, timeout: int = TIMEOUT, retries: int = RETRIES) -> dict:
    """GET and parse JSON, retrying with exponential backoff."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(BACKOFF ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def fetch_index(filter_id: str) -> list[int]:
    """Available weekly block timestamps for one series, ascending."""
    return sorted(get_json(cfg.index_url(filter_id)).get("timestamps", []))


def block_path(short: str, timestamp: int) -> Path:
    return cfg.raw_dir(short) / f"{timestamp}.json"


def cached_blocks(short: str) -> set[int]:
    d = cfg.raw_dir(short)
    if not d.exists():
        return set()
    return {int(p.stem) for p in d.glob("*.json")}


def read_block(short: str, timestamp: int) -> list:
    with block_path(short, timestamp).open(encoding="utf-8") as f:
        return json.load(f).get("series", [])


def ingest_series(short: str, dry_run: bool = False, verbose: bool = True) -> dict:
    """Fetch every missing block for one series and report what is on disk."""
    meta = cfg.SERIES[short]
    filter_id = meta["filter"]
    cfg.raw_dir(short).mkdir(parents=True, exist_ok=True)

    start_ms = int(cfg.WINDOW_START.timestamp() * 1000)
    wanted = [t for t in fetch_index(filter_id) if t >= start_ms]
    have = cached_blocks(short)
    stale = set(wanted[-ALWAYS_REFETCH_BLOCKS:])
    todo = [t for t in wanted if t not in have or t in stale]

    if verbose:
        fresh = len(have & set(wanted) - stale)
        print(f"  {short:<16} in window: {len(wanted):>4}   "
              f"cached: {fresh:>4}   to fetch: {len(todo):>4}   "
              f"(of which refreshed: {len(stale & have)})")

    if dry_run:
        return {"filter_id": filter_id, "role": meta["role"],
                "blocks_expected": len(wanted), "blocks_stored": len(have & set(wanted)),
                "fetched": 0, "failed": []}

    failed = []
    for i, ts in enumerate(todo, 1):
        try:
            payload = get_json(cfg.block_url(filter_id, ts))
        except Exception as exc:
            failed.append({"timestamp": ts, "error": repr(exc)})
            continue
        with block_path(short, ts).open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        time.sleep(POLITE_DELAY)
        if verbose and (i % 50 == 0 or i == len(todo)):
            print(f"    {short}: {i}/{len(todo)}")

    # Recount from disk so the report describes what is stored, not what the
    # loop believes it stored.
    stored = points = nulls = 0
    for ts in wanted:
        if not block_path(short, ts).exists():
            continue
        stored += 1
        series = read_block(short, ts)
        points += len(series)
        nulls += sum(1 for p in series if p[1] is None)

    return {
        "filter_id": filter_id,
        "role": meta["role"],
        "blocks_expected": len(wanted),
        "blocks_stored": stored,
        "fetched": len(todo) - len(failed),
        "failed": failed,
        "points": points,
        "nulls": nulls,
        "null_pct": round(100 * nulls / points, 4) if points else None,
        "first_block": min(wanted) if wanted else None,
        "last_block": max(wanted) if wanted else None,
    }


def ingest_all(dry_run: bool = False, verbose: bool = True) -> dict:
    """Ingest every registered series. Returns a manifest."""
    from datetime import datetime, timezone

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "window_start_utc": cfg.WINDOW_START.isoformat(),
        "region": cfg.REGION,
        "resolution": cfg.RESOLUTION,
        "series": {},
    }
    t0 = time.time()
    for short in cfg.ALL_SERIES:
        if verbose:
            print(f"\n[{short}] role={cfg.SERIES[short]['role']}")
        try:
            manifest["series"][short] = ingest_series(short, dry_run, verbose)
        except Exception as exc:
            manifest["series"][short] = {"error": repr(exc)}
            if verbose:
                print(f"  FAILED: {exc!r}")
    manifest["elapsed_seconds"] = round(time.time() - t0, 1)

    if not dry_run:
        cfg.DATA.mkdir(parents=True, exist_ok=True)
        with (cfg.DATA / "raw_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return manifest
