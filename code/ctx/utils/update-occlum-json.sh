#!/bin/sh

set -e

f1=$1
f2=$2
if ! { [ -f "$f1" ] && [ -f "$f2" ] ; }; then
    echo "Need two files" >&2
    exit 1
fi

set -u

tmp="$( mktemp )"
cat "$f1" "$f2" | jq -s 'reduce .[] as $item ({}; . * $item)' >"$tmp"
mv "$tmp" "$f1"
