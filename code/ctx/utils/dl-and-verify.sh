#!/bin/sh

set -u

die () {
    printf "%s\n" "$1" >&2
    exit 1
}

[ "$#" != 3 ] && die "Need url name hash"

url="$1"
name="$2"
sha256="$3"

wget "$url" -O "$name" ||
    die "Download failed"
( printf "%s  %s\n" "$sha256" "$name" | sha256sum -c ) ||
    die "Verification failed"
