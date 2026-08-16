#!/usr/bin/env bash
# Render a demo take, retrying seeds until the episode is worth showing, then encode to mp4.
#
# render_chase.py deliberately does not choose its own take -- it renders one episode and
# records the outcome in summary.json. Choosing is this script's job, and it exists because
# the alternative is shipping whichever episode luck dealt: the policy falls in roughly a
# third of episodes, so an unfiltered render is a coin flip on whether the demo shows a
# rescue or a faceplant.
#
# Re-rolling is stated in the benchmark write-up rather than hidden. It selects a
# *representative successful* episode, not a typical one, and the eval numbers are what
# describe typical behaviour.
#
# Usage:
#   ./make_demo.sh --checkpoint <path> --out ~/demo_v4 [--victim_radius 5.0 6.5]
#                  [--task ...] [--steps 900] [--tries 8]

set -euo pipefail

TASK="Isaac-Rescue-Anymal-C-Play-v0"
CHECKPOINT=""
OUT=""
STEPS=900
TRIES=8
RADIUS=""
NAV_CKPT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --tries) TRIES="$2"; shift 2 ;;
        --victim_radius) RADIUS="$2 $3"; shift 3 ;;
        --navigator_checkpoint) NAV_CKPT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "${CHECKPOINT}" ] && [ -n "${OUT}" ] || { echo "need --checkpoint and --out" >&2; exit 2; }

command -v ffmpeg >/dev/null 2>&1 || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg

mkdir -p "${OUT}"
BEST_DIR=""

for try in $(seq 1 "${TRIES}"); do
    # Seeds are arbitrary but reproducible, so a good take can be regenerated exactly.
    SEED=$((1000 + try * 137))
    DIR="${OUT}/take_${try}"
    rm -rf "${DIR}"; mkdir -p "${DIR}"
    echo "=== take ${try}/${TRIES}, seed ${SEED} ==="

    # shellcheck disable=SC2086
    ./isaaclab.sh -p harness/render_chase.py \
        --task "${TASK}" \
        --checkpoint "${CHECKPOINT}" \
        --num_envs 1 \
        --steps "${STEPS}" \
        --seed "${SEED}" \
        --out "${DIR}" \
        ${RADIUS:+--victim_radius ${RADIUS}} \
        ${NAV_CKPT:+--navigator_checkpoint "${NAV_CKPT}"} \
        --headless 2>&1 | grep -aE "navigator loaded|no navigator|victim spawn annulus|thermal panel|VOICE|Error|Traceback" || true

    [ -f "${DIR}/summary.json" ] || { echo "take ${try}: no summary.json, render failed"; continue; }
    FOUND=$(python3 -c "import json;print(json.load(open('${DIR}/summary.json'))['found_victim'])")
    FELL=$(python3 -c "import json;print(json.load(open('${DIR}/summary.json'))['fell'])")
    echo "take ${try}: found=${FOUND} fell=${FELL}"

    if [ "${FOUND}" = "True" ] && [ "${FELL}" = "False" ]; then
        BEST_DIR="${DIR}"; echo "=== clean take on try ${try} ==="; break
    fi
    # Keep the first found-but-fell take as a fallback: a rescue that ends in a stumble is a
    # far better demo than no rescue at all, and on a debris scene the policy has never
    # trained for, a perfectly clean take may simply not exist.
    if [ "${FOUND}" = "True" ] && [ -z "${BEST_DIR}" ]; then BEST_DIR="${DIR}"; fi
done

[ -n "${BEST_DIR}" ] || { echo "!!! no take found a casualty in ${TRIES} attempts" >&2; exit 1; }

# 50 fps = 1/step_dt (decimation 4 x sim dt 0.005), so the video runs at real time rather
# than the slow motion a default 25 fps would give.
ffmpeg -y -loglevel error -framerate 50 -i "${BEST_DIR}/f%05d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "${OUT}/demo.mp4"

cp "${BEST_DIR}/summary.json" "${OUT}/summary.json"
[ -f "${BEST_DIR}/voice_log.jsonl" ] && cp "${BEST_DIR}/voice_log.jsonl" "${OUT}/voice_log.jsonl"
# Frames are ~1 MB each and there can be 900 of them per take; the mp4 is the deliverable.
rm -rf "${OUT}"/take_*

echo
echo "wrote ${OUT}/demo.mp4"
cat "${OUT}/summary.json"
