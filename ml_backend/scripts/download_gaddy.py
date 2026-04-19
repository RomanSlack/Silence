"""Download the Gaddy & Klein 2020 silent-speech EMG dataset from Zenodo.

Zenodo record: https://zenodo.org/records/4064409  (DOI 10.5281/zenodo.4064408)

The dataset is ~5 GB. We stream the tar archive to data/raw/gaddy/ and extract.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "4064409"
API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/gaddy"))
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = requests.get(API, timeout=30).json()
    files = meta.get("files", [])
    if not files:
        print("No files in Zenodo metadata; check record ID.", file=sys.stderr)
        return 1

    for f in files:
        name = f["key"]
        url = f["links"]["self"]
        size = f.get("size", 0)
        dest = args.out / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"[skip] {name} already downloaded")
            continue

        print(f"[get ] {name}  ({size/1e9:.2f} GB)")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh, tqdm(total=size, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))

        if not args.skip_extract and name.endswith((".tar", ".tar.gz", ".tgz")):
            print(f"[tar ] extracting {name}")
            with tarfile.open(dest) as tf:
                tf.extractall(args.out)

    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
