#!/usr/bin/env bash
# Bring a bare Ubuntu 22.04 + NVIDIA GPU box up to the stack this project trains on.
#
# This exists because the original training host was configured by hand and then destroyed.
# Nothing in the repo recorded how, so rebuilding it meant reading version strings back out
# of artifacts/benchmarks/stage1.log. Do not let that happen twice: if you change the stack,
# change it here.
#
# The stack, recovered from the logs of the run that produced model_4197.pt:
#   Python 3.12          venv at ~/isaac/venv (extscache paths in the log read cp312)
#   Isaac Sim 6.0.1.0    pip, from pypi.nvidia.com, with the extscache extra
#   Isaac Lab 3.0.0-beta2  cloned to ~/isaac/IsaacLab (the log stores a git diff, so: source)
#   rsl-rl               plain site-package, pulled in by isaaclab.sh --install
#
# Paths are not free choices -- harness/stage2_search.sh hardcodes ~/isaac/venv and
# ~/isaac/IsaacLab. Change them here and you must change them there.
#
# Usage:  ./setup_isaac.sh          # on the target box
# Idempotent: every step short-circuits if it already ran, so re-run it after a failure.

set -euo pipefail

# Fail fast on a stalled socket instead of hanging on it. Learned the hard way: an install
# sat wedged for 95 minutes on one line with no pip process alive and no bytes moving,
# because pip's default has no ceiling on how long a dead connection may stall. These are
# exported rather than passed as flags so they also reach the pip calls nested inside
# isaaclab.sh --install, which is exactly where it hung.
export PIP_DEFAULT_TIMEOUT=60
export PIP_RETRIES=8

ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac}"
VENV="${ISAAC_ROOT}/venv"
ISAACLAB="${ISAAC_ROOT}/IsaacLab"
ISAACLAB_TAG="${ISAACLAB_TAG:-v3.0.0-beta2}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-6.0.1.0}"

step() { echo; echo "=== $* ==="; }

##
# 1. Vulkan gate -- run before anything expensive.
##
# The RTX renderer is not optional here: the pseudo-thermal camera IS a render pass, so no
# Vulkan means no thermal camera means no rescue task. RunPod was abandoned for exactly this
# -- its loader created an instance but reported no usable device. Checking it first turns a
# two-hour wasted install into a thirty-second failure.
step "Vulkan gate"
if ! command -v vulkaninfo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq vulkan-tools libvulkan1
fi
if ! vulkaninfo --summary 2>/dev/null | grep -q "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"; then
    echo "FATAL: no discrete-GPU Vulkan device." >&2
    echo "A CPU 'llvmpipe' device alone is not enough -- that is the RunPod failure mode." >&2
    vulkaninfo --summary 2>&1 | sed -n '/Devices:/,$p' >&2 || true
    exit 1
fi
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverInfo" | head -4

##
# 2. Runtime libraries Kit dlopen()s.
##
# libxt6 is the specific one worth naming: without it Kit throws three screens of
# "Could not load the dynamic library ... libMaterialXRenderHw.so: libXt.so.6" at every
# launch (see the head of artifacts/benchmarks/train.log). Non-fatal, but it buries real
# errors under noise, and a log you have stopped reading is a log that is not helping you.
step "Kit runtime libraries"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libglu1-mesa libgomp1 \
    libgl1 libegl1 libsm6 libice6 git curl

##
# 3. Python 3.12 via uv.
##
# The box ships 3.10; the stack is 3.12. uv fetches a managed CPython without a PPA and
# without sudo, which is both faster and less likely to disturb the system Python that other
# tooling on the image depends on.
#
# --seed is not optional: a bare `uv venv` omits pip, and isaaclab.sh --install shells out to
# pip directly rather than going through uv.
step "Python 3.12 venv at ${VENV}"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x "${VENV}/bin/python" ]; then
    mkdir -p "${ISAAC_ROOT}"
    uv venv --python 3.12 --seed "${VENV}"
fi
"${VENV}/bin/python" -V

##
# 4. Isaac Sim.
##
# The extscache extra is what the previous run used -- its Kit logs resolve extensions out of
# site-packages/isaacsim/extscache/... -- and it is the difference between shipping the
# extensions and downloading them on first launch. This is the long step: tens of GB.
step "Isaac Sim ${ISAACSIM_VERSION}"
if ! "${VENV}/bin/python" -c "import isaacsim" >/dev/null 2>&1; then
    "${VENV}/bin/pip" install --upgrade pip
    "${VENV}/bin/pip" install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com
fi

##
# 5. Isaac Lab from source.
##
# Source rather than pip because the training entry point is a script inside the tree
# (scripts/reinforcement_learning/rsl_rl/train.py) and because the task package gets dropped
# into isaaclab_tasks/ in step 6.
step "Isaac Lab ${ISAACLAB_TAG}"
if [ ! -d "${ISAACLAB}/.git" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB}"
fi
cd "${ISAACLAB}"
git fetch --tags --quiet
git checkout --quiet "${ISAACLAB_TAG}"

# isaaclab.sh picks its interpreter from an active VIRTUAL_ENV, so activate rather than
# calling the venv's python directly.
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh --install

step "Done"
echo "venv:     ${VENV}"
echo "isaaclab: ${ISAACLAB} @ $(git -C "${ISAACLAB}" describe --tags --always)"
echo
echo "Next: deploy_task.sh to install the rescue task package, then smoke_phase3.sh."
