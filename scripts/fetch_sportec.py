"""Download the Sportec/IDSSE open tracking data into the local cache.

The Sportec/IDSSE dataset (Bassek et al., 2025; CC-BY 4.0) is openly licensed but
large (about 2.6 GB of position XML across seven matches), so it is not committed to
the repository. This script fetches the seven matches' match-information and raw
position files from the PySport HuggingFace mirror used by kloppy into
data/cache/sportec/, so that scripts/analysis_fidelity_sportec.py can be re-run from
a single command. Downloads resume if interrupted (HTTP range requests), and files
already complete are skipped.

Usage:
    python scripts/fetch_sportec.py            # all seven matches
    python scripts/fetch_sportec.py J03WPY     # a subset by match id
"""
from __future__ import annotations

import sys
import urllib.request

from ffields import load_config, repo_root

BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
COMP = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
        "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002",
        "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
        "J03WR9": "DFL-COM-000002"}
TEMPLATES = {
    "meta": "DFL_02_01_matchinformation_{comp}_DFL-MAT-{mid}.xml",
    "tracking": "DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{mid}.xml",
}


def _complete(path):
    if not path.exists() or path.stat().st_size < 5000:
        return False
    with open(path, "rb") as fh:
        fh.seek(-200, 2)
        return b"</PutDataRequest>" in fh.read()


def _download(url, dest):
    """Resumable download with a Range header; appends to a partial file."""
    start = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url)
    if start:
        req.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "ab") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def main() -> None:
    cfg = load_config(); root = repo_root()
    sdir = root / cfg["paths"]["data_cache"] / "sportec"
    sdir.mkdir(parents=True, exist_ok=True)
    ids = sys.argv[1:] or list(COMP)
    for mid in ids:
        comp = COMP[mid]
        for kind, tmpl in TEMPLATES.items():
            fname = tmpl.format(comp=comp, mid=mid)
            dest = sdir / (f"meta_{mid}.xml" if kind == "meta" else f"track_{mid}.xml")
            if kind == "tracking" and _complete(dest):
                print(f"skip  {dest.name} (complete)")
                continue
            if kind == "meta" and dest.exists() and dest.stat().st_size > 5000:
                print(f"skip  {dest.name} (present)")
                continue
            print(f"fetch {dest.name} ...", flush=True)
            try:
                _download(f"{BASE}/{fname}", dest)
            except Exception as e:
                print(f"  WARNING: {type(e).__name__}: {e} (re-run to resume)")
    print("done")


if __name__ == "__main__":
    main()
