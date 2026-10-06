#!/usr/bin/env bash
# Download the two CaloChallenge ds2 files into Tina/ and check them.
#
#     bash cluster/get_data.sh
#
# Best run on the login machine: its own disk holds /home, so the 2.7 GB is written
# locally rather than over the network. Safe to rerun: a complete file is only
# checked, and a partial one resumes.
#
# Setting: DATA_DIR=/path/to/dir keeps the files there and links them into Tina/.

set -euo pipefail
cd "$(dirname "$0")/.."                 # the Tina/ folder

DATA_DIR=$(realpath -m "${DATA_DIR:-.}")
ZENODO=https://zenodo.org/records/6366271/files

# One run at a time, even across machines sharing /home (mkdir is atomic on
# NFS too), so two downloads never write the same file.
mkdir -p "$DATA_DIR"
lock="$DATA_DIR/.get_data.lock"
until mkdir "$lock" 2>/dev/null; do
    echo "$(date +%H:%M) waiting for another get_data.sh to finish (if none is running, remove $lock)"
    sleep 60
done
trap 'rmdir "$lock"' EXIT

# Size in bytes and MD5 as listed on the Zenodo record. The laptop copies that
# every result so far was computed from match them (checked 2026-09-13).
get_file() {    # file name, size in bytes, MD5
    local name=$1 size=$2 md5=$3
    local path="$DATA_DIR/$name"
    if [ "$(stat -L -c %s "$path" 2>/dev/null || echo 0)" != "$size" ]; then
        echo "downloading $name to $DATA_DIR"
        # -C - resumes a partial file; --retry rides out short outages.
        curl -L --fail -C - --retry 10 --retry-delay 30 -o "$path" "$ZENODO/$name?download=1"
    fi
    echo "checking MD5 of $name"
    if [ "$(md5sum < "$path" | cut -d' ' -f1)" != "$md5" ]; then
        echo "$name: MD5 does not match. Delete $path and rerun."
        exit 1
    fi
    if [ ! -e "$name" ] && [ ! -L "$name" ]; then
        ln -s "$path" "$name"           # the code reads the data from Tina/
    fi
    echo "$name: OK"
}

get_file dataset_2_1.hdf5 1356475617 e590333e9a2da51b258288d74bd8357a
get_file dataset_2_2.hdf5 1366708391 7a56fd68aa53ded37c2ac445694d9736
