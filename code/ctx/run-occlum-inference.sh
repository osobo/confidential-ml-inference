#!/bin/sh

die () {
    printf "%s\n" "$1" >&2
    exit 1
}

set -e
set -u

[ "$#" -ne 4 ] && die "Need 4 args"

server_timing="$1"
connections="$2"
batch_size="$3"
batches="$4"

/opt/occlum/start_aesm.sh

cd /root/occlum_workspace
occlum run /bin/inference-server "$server_timing" /root/model.onnx >../server-out &
sleep 10

cd /root

python3 inference-client.py \
    --data-path testdata.pkl.gz \
    --connections "$connections" \
    --batch-size "$batch_size" \
    --batches "$batches" \
    >client-out

python3 summarize-nums.py \
    --meta connections:"$connections" \
    --meta batch_size:"$batch_size" \
    --meta batches:"$batches" \
    --csv-file client:client-out \
    --csv-file server:server-out
