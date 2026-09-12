#!/usr/bin/env bash
# Fan out the DSV4.1 base image from the single pull origin (spark1) to the
# other two ranks, over the RoCE fabric. Fleet rule: pull once, fan out —
# never parallel-pull.
set -Eeuo pipefail
IMG=lmsysorg/sglang:dev-dsv41
KEY="$HOME/.ssh/id_ed25519_shared"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $KEY -o IdentitiesOnly=yes"
mkdir -p "$HOME/spark-runs"
LOG="$HOME/spark-runs/dsv41-image-fanout.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) waiting for initial pull to finish ==="
for _ in $(seq 1 360); do
  pgrep -f "docker pull" >/dev/null || break
  sleep 20
done
until docker image inspect "$IMG" >/dev/null 2>&1; do
  echo "$(date) image absent — re-pulling"
  docker pull --platform linux/arm64 "$IMG" || sleep 20
done
ID=$(docker image inspect "$IMG" --format '{{.Id}}')
echo "PULL_OK id=$ID size=$(docker image inspect "$IMG" --format '{{.Size}}')"

for t in spark2@10.73.0.2 spark3@10.73.0.3; do
  ip="${t#*@}"
  echo "=== $(date) fanout -> $t ==="
  docker save "$IMG" | $SSH "$t" 'docker load'
  rid=$($SSH "$t" "docker image inspect $IMG --format '{{.Id}}'" 2>/dev/null || echo MISSING)
  echo "target $ip id=$rid"
  if [ "$rid" != "$ID" ]; then echo "FANOUT_MISMATCH on $ip"; exit 1; fi
done
echo "IMAGE_FANOUT_DONE $(date)"
touch "$HOME/spark-runs/.dsv41-image-fanout.done"
