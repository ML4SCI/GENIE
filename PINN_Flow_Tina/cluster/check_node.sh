#!/usr/bin/env bash
# Report what this machine offers for the Tina/ setup. Changes nothing.
#
# Run it on each machine you might use (login node, GPU node):
#     bash cluster/check_node.sh
#
# What the setup needs:
#   python    python3.12 with a working venv module (on Ubuntu that is the
#             python3.12-venv package; without it, venv creation fails)
#   disk      about 10 GB: 2.7 GB of data plus a venv with CUDA torch
#   gpu       nvidia-smi's "CUDA Version" (the newest CUDA the driver
#             supports) must be at least the CUDA version torch was built for
#   internet  pip, git clone and the data download need pypi, github, zenodo

have() { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n== %s ==\n' "$1"; }

section "machine"
echo "host:  $(hostname)"
echo "os:    $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "glibc: $(ldd --version 2>/dev/null | head -1)"
echo "cpus:  $(nproc 2>/dev/null)"
free -g 2>/dev/null | head -2

section "home and disks"
# /home lives on the login machine; other machines mount it over NFS with Kerberos, so
# there it is only readable while you hold a valid ticket (see klist).
echo "home is on: $(findmnt -n -o SOURCE,FSTYPE --target "$HOME")"
klist 2>&1 | head -6
df -h -x tmpfs -x devtmpfs -x overlay -x squashfs -x efivarfs 2>/dev/null
have quota && timeout 10 quota -s 2>/dev/null

section "python"
for py in python3.12 python3; do
    if have "$py"; then
        # The only reliable test is making a throwaway venv (deleted again).
        tmp=$(mktemp -d)
        if "$py" -m venv "$tmp/venv" >/dev/null 2>&1; then venv=works; else venv=FAILS; fi
        rm -rf "$tmp"
        echo "$py: $(command -v "$py"), $("$py" --version 2>&1), creating a venv: $venv"
    fi
done
for tool in conda micromamba uv; do
    have "$tool" && echo "$tool: $(command -v "$tool")"
done

section "tools"
for tool in git curl wget tmux screen md5sum; do
    printf '%-7s %s\n' "$tool" "$(command -v "$tool" || echo MISSING)"
done

section "gpu"
if have nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
    nvidia-smi | grep -o 'CUDA Version: [0-9.]*'
    echo "processes using the GPU now:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
else
    echo "no nvidia-smi on this machine"
fi

section "internet"
for url in https://pypi.org https://github.com https://zenodo.org; do
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$url")
    echo "$url -> HTTP $code (000 = no connection)"
done
