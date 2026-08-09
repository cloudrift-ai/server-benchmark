#!/bin/bash
# Keep the CloudRift key out of the model's tool environment and retain only an
# unexported in-memory copy for the shell-trap VM cleanup after the agent exits.

set -euo pipefail

key_file=$1
shift
if [ "$(stat -c '%a' "$key_file")" != "600" ]; then
    echo "CloudRift API key file permissions must be 0600" >&2
    exit 1
fi
cloudrift_key=$(<"$key_file")
rm -f "$key_file"
unset CLOUDRIFT_API_KEY

cleanup_vm() {
    provider=$(python3 -c 'import json, os; print(json.load(open(os.environ["VM_LEASE"]))["vm"]["provider"])')
    if [ "$provider" = "gcp" ]; then
        echo "GCP cleanup deferred to the isolated always-cleanup step"
        return 0
    fi
    CLOUDRIFT_API_KEY="$cloudrift_key" ./venv/bin/emmy vm delete lease "$VM_LEASE" \
        --owner "$GITHUB_REPOSITORY/$RUN_OWNER"
}
trap cleanup_vm EXIT INT TERM

exec 3<<<"$cloudrift_key"
./venv/bin/emmy agent run --api-key-fd 3 "$@"
exec 3<&-
