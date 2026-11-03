"""Download Teeth3DS+ from OSF.

Uses the same OSF links as torch_geometric.datasets.Teeth3DS, but lets you
download only the parts you need. Each part contains .obj meshes with their
.json label files next to them.

Interrupted downloads are resumed: partial data is kept in <name>.zip.part,
and re-running the script continues from where it stopped.

Examples:
    python download_data.py --sample          # small sample, good for a first test
    python download_data.py --parts 1         # one data part
    python download_data.py --parts 1 2 --splits
"""
import argparse
import http.client
import re
import time
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

PARTS = {
    1: "https://osf.io/download/qhprs/",
    2: "https://osf.io/download/4pwnr/",
    3: "https://osf.io/download/frwdp/",
    4: "https://osf.io/download/2arn4/",
    5: "https://osf.io/download/xrz5f/",
    6: "https://osf.io/download/23hgq/",
    7: "https://osf.io/download/u83ad/",
}
SAMPLE_URL = "https://osf.io/download/vr38s/"
# Zip of the OSF project's root storage (holds the official train/test split files).
SPLITS_URL = "https://files.de-1.osf.io/v1/resources/xctdy/providers/osfstorage/?zip="
SPLITS_MAX_BYTES = 50 * 1024**2  # abort if this is unexpectedly large


class TooLarge(RuntimeError):
    pass


def download(url: str, dest: Path, max_bytes: int | None = None, retries: int = 20) -> None:
    """Download url to dest, resuming from dest.part after interruptions."""
    part = dest.with_name(dest.name + ".part")
    total = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the range request: start over
                    have = 0
                content_range = r.headers.get("Content-Range")  # e.g. "bytes 100-999/1000"
                length = int(r.headers.get("Content-Length") or 0)
                if content_range and (m := re.search(r"/(\d+)$", content_range)):
                    total = int(m.group(1))
                elif length:
                    total = have + length
                if max_bytes and total and total > max_bytes:
                    raise TooLarge(f"{dest.name} is {total / 1e6:.0f} MB, larger than expected")

                done = have
                with open(part, "ab" if have else "wb") as f, \
                        tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dest.name) as bar:
                    while chunk := r.read(1 << 20):
                        done += len(chunk)
                        if max_bytes and done > max_bytes:
                            raise TooLarge(f"{dest.name} exceeded {max_bytes / 1e6:.0f} MB")
                        f.write(chunk)
                        bar.update(len(chunk))

            size = part.stat().st_size
            if total is None or size >= total:
                part.replace(dest)
                return
            print(f"Connection closed at {size / 1e6:.0f}/{total / 1e6:.0f} MB, resuming...")
        except TooLarge:
            part.unlink(missing_ok=True)
            raise
        except (OSError, http.client.HTTPException) as e:
            print(f"Interrupted ({e}); retrying in 5 s (attempt {attempt}/{retries})")
            time.sleep(5)
    raise RuntimeError(f"{dest.name}: download incomplete after {retries} attempts. "
                       "Run the script again to resume.")


def fetch(name: str, url: str, out: Path, max_bytes: int | None = None) -> None:
    marker = out / f".{name}.done"
    if marker.exists():
        print(f"{name}: already downloaded, skipping")
        return
    zip_path = out / f"{name}.zip"
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        download(url, zip_path, max_bytes)
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
        raise RuntimeError(f"{zip_path.name} is not a valid zip; deleted it. Run the script again.")
    print(f"{name}: extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    zip_path.unlink()
    marker.touch()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parts", type=int, nargs="*", default=[], choices=sorted(PARTS),
                   help="data parts to download (1-7)")
    p.add_argument("--sample", action="store_true", help="download the small sample")
    p.add_argument("--splits", action="store_true", help="download official train/test split files")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    args = p.parse_args()

    if not (args.parts or args.sample or args.splits):
        p.error("nothing to download: use --sample, --parts and/or --splits")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.sample:
        fetch("teeth3ds_sample", SAMPLE_URL, args.out)
    for i in args.parts:
        fetch(f"data_part_{i}", PARTS[i], args.out)
    if args.splits:
        try:
            fetch("train_test_split", SPLITS_URL, args.out, SPLITS_MAX_BYTES)
        except TooLarge as e:
            print(f"Splits not downloaded: {e}")

    uppers = list(args.out.rglob("*_upper.obj"))
    labelled = [o for o in uppers if o.with_suffix(".json").exists()]
    print(f"\nFound {len(uppers)} upper-jaw meshes, {len(labelled)} with labels, in {args.out}")


if __name__ == "__main__":
    main()