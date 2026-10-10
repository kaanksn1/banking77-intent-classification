#!/usr/bin/env bash
# Ubuntu 24.04 WSL only. Run as root after Windows/WSL installation is complete.
# Sources and platform requirements: docs/NEURAL_BASELINES.md.
set -euo pipefail
umask 022

if [[ $(id -u) != 0 ]]; then
    printf '%s\n' 'Run this setup with sudo; training commands only need the resulting venv.' >&2
    exit 1
fi
if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
    printf '%s\n' 'This setup is for WSL, not native Linux.' >&2
    exit 1
fi
source /etc/os-release
if [[ $ID != ubuntu || $VERSION_ID != 24.04 ]]; then
    printf '%s\n' 'This recipe requires Ubuntu 24.04 and Python 3.12.' >&2
    exit 1
fi

repo_directory=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
download_directory=/opt/banking77-downloads
venv_directory=/opt/banking77-venv
mkdir -p "$download_directory"
export DEBIAN_FRONTEND=noninteractive

printf '%s\n' 'Installing Ubuntu prerequisites and the pinned AMD package repositories...'
apt-get update
apt-get install -y wget ca-certificates git python3.12-venv python3-setuptools python3-wheel
installer=amdgpu-install_7.2.1.70201-1_all.deb
wget -q --https-only --timeout=60 --tries=3 -O "$download_directory/$installer" \
    "https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/$installer"
apt-get install -y "$download_directory/$installer"
apt-get update

# WSL uses the Windows display driver. Do not install amdgpu-dkms/Linux kernels.
printf '%s\n' 'Installing ROCm 7.2.1 userspace...'
apt-cache policy rocm
rocm_version=$(apt-cache policy rocm | awk '/Candidate:/ {print $2}')
if [[ $rocm_version != 7.2.1.* ]]; then
    printf 'Expected ROCm 7.2.1, found repository candidate %s\n' "$rocm_version" >&2
    exit 1
fi
apt-get install -y "rocm=$rocm_version"

printf '%s\n' 'Installing the versioned ROCDXG WSL bridge...'
bridge=rocdxg-roct_1.2.0_amd64.deb
wget -q --https-only --timeout=60 --tries=3 -O "$download_directory/$bridge" \
    "https://github.com/ROCm/librocdxg/releases/download/v1.2.0/$bridge"
printf '%s  %s\n' '3ed9526719290cd8f590150dad8ea0f234fa779bea6a4c9a8449d7ae6b8cfb6e' \
    "$download_directory/$bridge" | sha256sum -c -
dpkg -i "$download_directory/$bridge"
export HSA_ENABLE_DXG_DETECTION=1
export PATH="/opt/rocm/bin:$PATH"
rocminfo

printf '%s\n' 'Preparing the separate Linux Python environment...'
python3.12 -m venv "$venv_directory"
python_command="$venv_directory/bin/python"
"$python_command" -m pip install --upgrade pip wheel
cd "$repo_directory"
"$python_command" -m pip install -r requirements-lock.txt

torch_wheel='torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl'
triton_wheel='triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'
wheel_base='https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1'
printf '%s\n' 'Downloading the official AMD PyTorch and Triton wheels...'
wget -q --https-only --timeout=60 --tries=3 -c -O "$download_directory/$torch_wheel" "$wheel_base/$torch_wheel"
wget -q --https-only --timeout=60 --tries=3 -c -O "$download_directory/$triton_wheel" "$wheel_base/$triton_wheel"
# AMD publishes these filenames/sizes; these locally computed hashes are provenance,
# not independent vendor checksums. The bridge checksum above IS vendor-published.
sha256sum "$download_directory/$installer" "$download_directory/$torch_wheel" \
    "$download_directory/$triton_wheel" > "$download_directory/download-sha256.txt"
"$python_command" -m pip install "$download_directory/$torch_wheel" "$download_directory/$triton_wheel"
"$python_command" -m pip install -r requirements-neural.txt
"$python_command" -m pip check
printf '%s\n' 'Checking a real GPU forward/backward pass...'
"$python_command" -m banking77.neural_preflight --require-gpu
printf '%s\n' 'Setup completed. Set HSA_ENABLE_DXG_DETECTION=1 in every training process.'
