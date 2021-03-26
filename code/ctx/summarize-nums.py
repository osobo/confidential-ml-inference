import argparse
import csv
import json
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", action="append", type=str)
    parser.add_argument("--meta", action="append", type=str)
    args = parser.parse_args()

    arrays = dict()
    for csv_arg in args.csv_file:
        parts = csv_arg.split(":")
        assert(len(parts) == 2)
        name = parts[0]
        path = parts[1]
        arrays[name] = file2nums(path)
    means = {
        name: { k: v.mean() for (k, v) in array.items() }
        for (name, array) in arrays.items()
    }

    meta_out = dict()
    for meta_arg in args.meta:
        parts = meta_arg.split(":")
        assert(len(parts) == 2)
        k = parts[0]
        v = parts[1]
        meta_out[k] = v

    output = dict(meta=meta_out, mean_times=means)

    print(json.dumps(output, indent=4))

def file2nums(filename):
    lists = dict()
    with open(filename, mode="rt") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            for k, v in row.items():
                if k not in lists:
                    lists[k] = []
                lists[k].append(float(v))
    return { name: np.array(lst) for name, lst  in lists.items() }

if __name__ == "__main__":
    main()
