#!/bin/sh

die () {
    printf "%s\n" "$1" >&2
    exit 1
}

set -e
set -u

[ "$#" -ne 5 ] && die "Need 5 args"

model_data_name="$1"
server_timing="$2"
connections="$3"
batch_size="$4"
batches="$5"

base=/root/models_and_data/"$model_data_name"
[ -d "$base" ] || die "No dir $base"
model="$base"/model.onnx
[ -f "$model" ] || die "No model $model"
data="$base"/testdata.pkl.gz
[ -f "$data" ] || die "No data $data"

cd /root

./inference-server "$server_timing" "$model" >server-out &
server_pid=$!
sleep 2

python3 inference-client.py \
    --data-path "$data" \
    --connections "$connections" \
    --batch-size "$batch_size" \
    --batches "$batches" \
    >client-out

kill "$server_pid"

python3 summarize-nums.py \
    --meta connections:"$connections" \
    --meta batch_size:"$batch_size" \
    --meta batches:"$batches" \
    --csv-file client:client-out \
    --csv-file server:server-out
