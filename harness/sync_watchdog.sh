#!/usr/bin/env bash
# Pull checkpoints off the training host on a loop, so a reclaimed instance costs minutes.
#
# This exists because of measured behaviour, not paranoia: two massedcompute L40S instances
# were reclaimed within twelve hours, the second one mid-run, taking a 30 GB install and an
# in-flight eval with it. Cloud checkpoints are not a backup -- they live on the machine most
# likely to disappear.
#
# Pairs with save_interval=25 in agents/rsl_rl_ppo_cfg.py. That bounds what the *host* loses
# between writes; this bounds what *you* lose between hosts.
#
# Deliberately a loop around sync_artifacts.sh rather than new sync logic -- that script
# already has the rsync filters and the host override, and two implementations of "pull the
# artifacts" would drift.
#
# Usage:
#   HOST=<instance> ./sync_watchdog.sh              # every 30 min until stopped
#   HOST=<instance> INTERVAL=600 ./sync_watchdog.sh # every 10 min
#
# Run it in the background on the laptop; stop it with Ctrl-C or by killing the pid.

set -uo pipefail   # NOT -e: a failed sync must not kill the watchdog

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:?set HOST to the training instance name}"
INTERVAL="${INTERVAL:-1800}"

echo "[watchdog] host=${HOST} interval=${INTERVAL}s"

while true; do
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    # A sync failing is the normal case when the box is gone, which is exactly when this
    # script must keep running rather than exit -- the host may come back, and either way the
    # last successful pull is already safe on disk.
    # sync_artifacts.sh lives at the REPO ROOT, not in harness/. Getting this wrong cost
    # three silent sync cycles -- the watchdog dutifully reported "host unreachable?" when
    # the host was fine and the script path was simply wrong. Worth noting the guess in the
    # error message was misleading: a failed sync has more causes than a dead host.
    if HOST="${HOST}" "${HERE}/../sync_artifacts.sh" >/tmp/sync_watchdog_last.log 2>&1; then
        newest="$(find "${HERE}/../artifacts/checkpoints" -name 'model_*.pt' -newermt '-35 minutes' 2>/dev/null | wc -l)"
        echo "[watchdog ${ts}] ok -- ${newest} checkpoint(s) new in the last 35 min"
    else
        echo "[watchdog ${ts}] SYNC FAILED (host unreachable?) -- see /tmp/sync_watchdog_last.log"
    fi
    sleep "${INTERVAL}"
done
