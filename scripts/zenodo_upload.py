"""
One-shot uploader for OpenCure prospective snapshots to Zenodo.

Usage:
    export ZENODO_TOKEN=<your personal access token>
    python3 scripts/zenodo_upload.py data/prospective/snapshots/<timestamp>/

What it does:
    1. Reads the snapshot's zenodo_metadata.json
    2. Creates a new Zenodo deposit (draft)
    3. Uploads predictions.jsonl + methods.json + README.md
    4. Publishes the deposit (immutable — can't be edited after)
    5. Prints the DOI and writes it into the snapshot's README

Sandbox mode: pass --sandbox to test against zenodo.sandbox.org (no real DOI
issued, but full API round-trip verified). Use this before going live.

Requires: requests (`pip install requests`)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def zenodo_base(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"


def upload(snapshot_dir: Path, token: str, sandbox: bool = False) -> dict:
    try:
        import requests
    except ImportError:
        raise SystemExit("requests not installed: pip install requests")

    meta_file = snapshot_dir / "zenodo_metadata.json"
    if not meta_file.exists():
        raise SystemExit(f"{meta_file} missing; regenerate snapshot.")
    meta = json.loads(meta_file.read_text())

    base = zenodo_base(sandbox)
    headers = {"Content-Type": "application/json"}
    params = {"access_token": token}

    # 1. Create empty deposit
    print("Creating deposit draft…")
    r = requests.post(f"{base}/api/deposit/depositions",
                      params=params, json={}, headers=headers)
    r.raise_for_status()
    dep = r.json()
    deposition_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    print(f"  deposition_id = {deposition_id}")

    # 2. Upload each file listed in metadata
    for filename in meta.get("files_to_upload", []):
        fp = snapshot_dir / filename
        if not fp.exists():
            print(f"  SKIP missing: {filename}")
            continue
        print(f"  uploading {filename} ({fp.stat().st_size:,} bytes)…")
        with fp.open("rb") as fh:
            r = requests.put(f"{bucket_url}/{filename}",
                             data=fh, params=params)
            r.raise_for_status()

    # 3. Add metadata (title, authors, description, license)
    payload = {"metadata": meta["metadata"]}
    r = requests.put(f"{base}/api/deposit/depositions/{deposition_id}",
                     params=params, json=payload, headers=headers)
    r.raise_for_status()

    # 4. Publish — this mints the DOI and locks the record.
    print("Publishing (DOI mint)…")
    r = requests.post(f"{base}/api/deposit/depositions/{deposition_id}/actions/publish",
                      params=params)
    r.raise_for_status()
    published = r.json()
    doi = published.get("doi") or published.get("metadata", {}).get("doi", "")
    url = published.get("links", {}).get("record_html", "")
    print(f"  DOI: {doi}")
    print(f"  URL: {url}")

    # 5. Write DOI back into the snapshot's README
    readme = snapshot_dir / "README.md"
    if readme.exists() and doi:
        text = readme.read_text()
        if doi not in text:
            text += f"\n\n## Zenodo DOI\n\n**{doi}**  \n{url}\n"
            readme.write_text(text)
            print(f"  updated {readme} with DOI")

    return {"deposition_id": deposition_id, "doi": doi, "url": url}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot_dir", type=Path)
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN env var not set. Generate one at "
                 "https://zenodo.org/account/settings/applications/tokens/new/")

    if not args.snapshot_dir.exists():
        sys.exit(f"{args.snapshot_dir} does not exist.")

    result = upload(args.snapshot_dir, token, sandbox=args.sandbox)
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
