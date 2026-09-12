#!/usr/bin/env python3
"""Measure raw Docker Hub blob-CDN throughput (anonymous token) to see whether
the ~1.4 MB/s docker-pull rate is a client/daemon artifact or the CDN itself."""
import json
import time
import urllib.request

REPO = "lmsysorg/sglang"
TAG = "dev-dsv41"


def get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


token = json.loads(
    get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull")[0]
)["token"]
auth = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json",
}
raw, _ = get(f"https://registry-1.docker.io/v2/{REPO}/manifests/{TAG}", auth)
doc = json.loads(raw)

digest = None
if "manifests" in doc:  # multi-arch index
    for m in doc["manifests"]:
        p = m.get("platform", {})
        if p.get("architecture") == "arm64" and p.get("os") == "linux":
            digest = m["digest"]
            print("arm64 manifest:", digest)
            break
    raw, _ = get(f"https://registry-1.docker.io/v2/{REPO}/manifests/{digest}", auth)
    doc = json.loads(raw)

layers = sorted(doc.get("layers", []), key=lambda x: -x["size"])
print(f"layers: {len(layers)}, total compressed: {sum(l['size'] for l in layers)/1e9:.2f} GB")
big = layers[0]
print(f"largest layer: {big['size']/1e6:.0f} MB")

# Pull a 150 MB range from the largest layer and time it.
limit = 150_000_000
req = urllib.request.Request(
    f"https://registry-1.docker.io/v2/{REPO}/blobs/{big['digest']}",
    headers={"Authorization": f"Bearer {token}", "Range": f"bytes=0-{limit-1}"},
)
t0 = time.time()
got = 0
with urllib.request.urlopen(req, timeout=60) as r:
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        got += len(chunk)
        if got >= limit or time.time() - t0 > 30:
            break
el = time.time() - t0
print(f"CDN blob test: {got/1e6:.1f} MB in {el:.1f}s = {got/el/1e6:.2f} MB/s")
