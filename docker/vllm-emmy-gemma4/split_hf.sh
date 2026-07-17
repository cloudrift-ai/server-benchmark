#!/bin/bash
# Split warm/hf into warm/hf_parts/p0..p3 for the bake. Docker Hub rejects very large
# blobs (the monolithic ~24 GB snapshot layer 503s at upload initiation, forever), so
# the Dockerfile COPYs four sub-10 GB parts that merge back into /opt/emmy/hf. Split is
# by hardlink (no extra disk); symlinks (the hub snapshots/ -> blobs/ links) stay
# symlinks. Everything except the big blobs/ payloads goes to p0; the blob files are
# balanced across all four parts, largest first.
set -euo pipefail
cd "$(dirname "$0")"
SRC=warm/hf
DST=warm/hf_parts

rm -rf "$DST"
mkdir -p "$DST"/p0 "$DST"/p1 "$DST"/p2 "$DST"/p3

# p0 = the full tree minus the blob payloads (configs, tokenizer, refs, symlinks)
cp -al "$SRC"/. "$DST"/p0/
find "$DST"/p0 -path "*/blobs/*" -type f -delete

# balance the blob payloads across p0..p3, largest first into the emptiest part
declare -a used=(0 0 0 0)
while read -r sz f; do
    rel=${f#"$SRC"/}
    best=0
    for i in 1 2 3; do [ "${used[$i]}" -lt "${used[$best]}" ] && best=$i; done
    mkdir -p "$DST/p$best/$(dirname "$rel")"
    ln "$f" "$DST/p$best/$rel"
    used[$best]=$((used[$best] + sz))
done < <(find "$SRC" -path "*/blobs/*" -type f -printf "%s %p\n" | sort -rn)

echo "[split] $(du -sh "$DST"/p0 "$DST"/p1 "$DST"/p2 "$DST"/p3 | tr '\n' ' ')"
