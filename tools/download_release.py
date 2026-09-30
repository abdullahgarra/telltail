#!/usr/bin/env python3
"""Download the tier-1 release bundle (retrievals + passages + score caches) into
`$TELLTAIL_DATA_DIR`, so `evaluate`/`sweep` can run from a fresh clone with no GPU.

The bundle URL is read from `$TELLTAIL_RELEASE_URL` (a `.tar.gz`/`.zip` of the
`generic/<query_set>/...` layout produced by tools/package_release.py). No URL is
hardcoded — set it once the bundle is hosted.

    TELLTAIL_RELEASE_URL=https://.../telltail_release.tar.gz \
    TELLTAIL_DATA_DIR=/path/to/data python tools/download_release.py
"""
import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def main() -> None:
    url = os.environ.get("TELLTAIL_RELEASE_URL")
    if not url:
        sys.exit("TELLTAIL_RELEASE_URL is not set. Point it at the hosted bundle "
                 "(.tar.gz/.zip). The bundle is built by tools/package_release.py; "
                 "host it and set this variable, then re-run.")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from telltail.paths import data_dir
    dest = data_dir()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[download] {url}\n[download] -> {dest}")
    with tempfile.NamedTemporaryFile(suffix=Path(url).suffix or ".bin", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        archive = tmp.name
    if archive.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        sys.exit(f"unsupported archive type: {archive}")
    os.unlink(archive)
    print(f"[download] extracted into {dest} "
          f"(expect generic/<query_set>/{{retrieval,passages,scores}}/)")


if __name__ == "__main__":
    main()
