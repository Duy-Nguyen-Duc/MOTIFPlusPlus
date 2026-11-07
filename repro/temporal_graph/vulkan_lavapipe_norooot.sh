#!/bin/bash
# Software Vulkan renderer (Mesa lavapipe) without root, as used for the reference runs: downloads the Ubuntu 22.04
# packages, unpacks them under <dir>/root and writes <dir>/root/lvp_icd.json with an absolute library path.
#   bash repro/temporal_graph/vulkan_lavapipe_norooot.sh <dir>      then      export VK_FALLBACK_ROOT=<dir>/root
# Needs apt-get (Ubuntu / Debian host or container); on other distributions install libvulkan1 + mesa-vulkan-drivers
# with the system package manager and point VK_ICD_FILENAMES at its lvp_icd.x86_64.json instead.
set -eu
DIR=${1:?usage: $0 <dir>}
mkdir -p "$DIR/debs" "$DIR/root"; cd "$DIR/debs"
PKGS="libvulkan1 mesa-vulkan-drivers libllvm15 libdrm2 libelf1 libzstd1 libedit2 libtinfo6 libbsd0 libmd0 libffi8 libx11-6 libx11-xcb1 libxcb1 libxcb-dri2-0 libxcb-dri3-0 libxcb-glx0 libxcb-present0 libxcb-randr0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxau6 libxdmcp6 libxshmfence1 libwayland-client0"
apt-get download $PKGS
for d in *.deb; do dpkg -x "$d" ../root; done
cd ..
ICD=$(find root/usr/share/vulkan/icd.d -name 'lvp_icd*.json' | head -1)
LIB=$(find root -name 'libvulkan_lvp.so' | head -1)
[ -n "$ICD" ] && [ -n "$LIB" ] || { echo "lavapipe files not found after unpacking"; exit 1; }
python3 - "$ICD" "$LIB" "$PWD/root/lvp_icd.json" <<'PY'
import json, sys, os
icd = json.load(open(sys.argv[1])); icd["ICD"]["library_path"] = os.path.abspath(sys.argv[2])
json.dump(icd, open(sys.argv[3], "w"), indent=4); print("wrote", sys.argv[3])
PY
echo "export VK_FALLBACK_ROOT=$PWD/root"
