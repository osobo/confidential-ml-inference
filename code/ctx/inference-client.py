import argparse
import gzip
import io
import logging
import multiprocessing as mp
import numpy as np
import pickle
import socket
import sys

from time import time
from logging import debug, info

PORT = 31648

CSV_HEADER = None

def main():
    conf_logger(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--connections", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batches", type=int)

    args = parser.parse_args()

    xs = None
    ys = None
    with gzip.open(args.data_path, "rb") as f:
        xs, ys, ncat = pickle.load(f)
    info(f"Read xs:{xs.shape} and ys:{ys.shape}")
    assert(xs.shape[0] == ys.shape[0])

    for _ in range(args.connections):
        one_run(ncat, xs, ys, args.batch_size, args.batches)

def one_run(ncat, xs, ys, batch_size, batches):
    server_address = ("localhost", PORT)

    idxs_for_each_batch = [
        np.random.choice(
            int(xs.shape[0]),
            batch_size,
            replace=True
        )
        for _ in range(batches)
    ]

    batch_time = None
    send_time = None
    recv_time = None
    conn_beg = time()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(server_address)
        info(f"Connected to server")
        bt, st, rt = with_conn(ncat, xs, ys, batch_size, idxs_for_each_batch, s)
        batch_time = bt
        send_time = st
        recv_time = rt
    conn_end = time()

    log_line(
        batch=batch_time,
        conn=(conn_end - conn_beg),
        send=send_time,
        recv=recv_time
    )

def log_line(**kwargs):
    global CSV_HEADER
    if CSV_HEADER is None:
        CSV_HEADER = list(kwargs.keys())
        print(",".join(CSV_HEADER))
    else:
        assert(set(CSV_HEADER) == set(kwargs))
    values = [ str(kwargs[key]) for key in CSV_HEADER ]
    print(",".join(values), flush=True)

def with_conn(ncat, xs, ys, batch_size, idxs_for_each_batch, sock):
    send_f = sock.makefile(mode="wb", buffering=0)
    recv_f = sock.makefile(mode="rb", buffering=0)

    send_f.write(int2bytes(batch_size))

    tot_conn_time = 0.0
    tot_send_time = 0.0
    tot_recv_time = 0.0
    tot_hits = 0

    for req_num, batch_idxs in enumerate(idxs_for_each_batch):
        assert(len(batch_idxs) == batch_size)

        buf = io.BytesIO()
        construct_request(req_num, xs, batch_idxs, buf)

        send_time = mp.Value('d', 0.0)
        send_beg = mp.Value('d', 0.0)
        recv_time = mp.Value('d', 0.0)
        recv_end = mp.Value('d', 0.0)
        hits = mp.Value('i', 0)

        send_p = mp.Process(
            target=send_request,
            args=(buf.getbuffer(), send_f, send_time, send_beg)
        )
        recv_p = mp.Process(
            target=handle_respose,
            args=(req_num, ncat, ys, batch_idxs, recv_f, recv_time, recv_end, hits)
        )

        send_p.start()
        recv_p.start()

        send_p.join()
        assert(send_p.exitcode == 0)
        recv_p.join()
        assert(recv_p.exitcode == 0)

        buf.close()

        tot_conn_time += recv_end.value - send_beg.value
        tot_send_time += send_time.value
        tot_recv_time += recv_time.value
        tot_hits += hits.value

    tot_asks = batch_size * len(idxs_for_each_batch)
    info(f"hitrate {100*tot_hits/tot_asks:.2f}%")
    
    return tot_conn_time, tot_send_time, tot_recv_time

def construct_request(req_num, xs, idxs, f):
    f.write(int2bytes(req_num)) # send request id
    for i in idxs:
        x = xs[i]
        f.write(xs[i].tobytes())

def send_request(req_bytes, f, send_time, send_beg):
    beg = time()
    f.write(req_bytes)
    end = time()
    send_time.value = end - beg
    send_beg.value = beg

def handle_respose(req_num, ncat, ys, idxs, f, recv_time, recv_end, hits):
    elem_size = 4 # TODO: hardcoded for float32
    bytes_per_output = ncat * elem_size

    batch_size = len(idxs)

    to_read = 0
    to_read += 4 # request id
    to_read += bytes_per_output * batch_size

    beg = time()
    response = read_exact(f, to_read)
    end = time()

    hit_count = 0

    assert(bytes2int(response[:4]) == req_num)
    outputs_buf = response[4:]
    for idx_in_batch, data_idx in enumerate(idxs):
        beg_byte = idx_in_batch * bytes_per_output
        end_byte = beg_byte + bytes_per_output
        body = outputs_buf[beg_byte:end_byte]

        # TODO: hardcoded dtype
        model_scores = np.frombuffer(body, dtype=np.float32).reshape((ncat,))
        model_y = np.argmax(model_scores)

        correct_y = ys[data_idx]
        if model_y == correct_y:
            hit_count += 1

    recv_time.value = end - beg
    recv_end.value = end
    hits.value = hit_count

def read_exact(f, nbytes):
    buf = bytearray(nbytes)
    if nbytes == 0:
        return buf
    view = memoryview(buf)
    totread = 0
    while totread < nbytes:
        recv_buf = view[totread:]
        read_now = f.readinto(recv_buf)
        if read_now == 0:
            return None
        assert(read_now > 0)
        totread += read_now
    return buf

def bytes2int(bytes):
    assert(len(bytes) == 4)
    return int.from_bytes(bytes, byteorder="big", signed=False)

def int2bytes(n):
    return n.to_bytes(4, byteorder="big", signed="False")

def conf_logger(level):
    fmt = f"log:CLIENT:%(levelname)s    %(message)s"
    logging.basicConfig(stream=sys.stderr, level=level, format=fmt)

if __name__ == "__main__":
    main()
