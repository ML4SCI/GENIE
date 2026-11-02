#!/usr/bin/env bash
set -e

data_dir="${1:-$(dirname "$0")/../data/raw}"
if (( $# > 0 )); then
    shift
fi
datasets=("$@")
if (( ${#datasets[@]} == 0 )); then
    datasets=(LHCO extra_qcd BB1 XtoWRto3W XtoYYprime ZpToTpTp YtoHHto4T)
fi
mkdir -p "$data_dir"/{lhco,bb1,kitchensink}

download() {
    if [[ -f "$2" ]]; then
        printf 'Already exists: %s\n' "$2"
        return
    fi
    curl -fL "$1" -o "$2.part"
    mv "$2.part" "$2"
}

lhco=https://zenodo.org/records/6466204/files
extra=https://zenodo.org/records/8370758/files
# Published subjettiness tables (Git LFS file contents, not pointer files).
subjettiness=https://media.githubusercontent.com/media/uhh-pd-ml/treebased_anomaly_detection/main/dataset
bb1=https://zenodo.org/records/4536624/files
kitchensink=https://zenodo.org/records/18983506/files

for dataset in "${datasets[@]}"; do
    case "$dataset" in
        LHCO)
            download "$lhco/events_anomalydetection_v2.h5" \
                "$data_dir/lhco/events_anomalydetection.h5"
            download "$subjettiness/events_anomalydetection_v2.extratau_2.features.h5" \
                "$data_dir/lhco/lhco_subjettiness.h5"
            ;;
        extra_qcd)
            download "$extra/events_anomalydetection_qcd_extra_inneronly_4vecs_and_features.h5" \
                "$data_dir/lhco/extra_qcd_cartesian.h5"
            download "$extra/events_anomalydetection_qcd_extra_inneronly_features.h5" \
                "$data_dir/lhco/extra_qcd_features.h5"
            download "$subjettiness/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5" \
                "$data_dir/lhco/extra_qcd_subjettiness.h5"
            ;;
        BB1)
            for name in events_LHCO2020_BlackBox1.h5 events_LHCO2020_BlackBox1.masterkey; do
                download "$bb1/$name" "$data_dir/bb1/$name"
            done
            ;;
        XtoWRto3W|XtoYYprime|ZpToTpTp|YtoHHto4T)
            # Keep the original archive; extract only the two files we use.
            archive="$data_dir/kitchensink/$dataset.tar.gz"
            directory="$data_dir/kitchensink/$dataset"
            download "$kitchensink/$dataset.tar.gz" "$archive"
            mkdir -p "$directory"
            if [[ ! -f "$directory/events.h5" || ! -f "$directory/${dataset}_subjettinesses.h5" ]]; then
                tar -xzf "$archive" -C "$directory" \
                    ./events.h5 "./${dataset}_subjettinesses.h5"
            fi
            ;;
        *) printf 'Unknown dataset: %s\n' "$dataset" >&2; exit 2 ;;
    esac
done
