import gzip
import numpy as np
import onnx
from onnx.numpy_helper import to_array
import pickle

NCAT = 1000

def file2np(filename, skip, shape):
    with gzip.open(filename, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=skip).reshape(shape)

def read_pb(path):
    r = onnx.TensorProto()
    with open(path, mode="rb") as f:
        r.ParseFromString(f.read())
    return r

def read_dataset(path):
    inp = to_array(read_pb(f"{path}/input_0.pb"))
    out = to_array(read_pb(f"{path}/output_0.pb"))
    x = inp
    y = np.argmax(out)
    return x, y

xs = []
ys = []
for i in range(3):
    x, y = read_dataset(f"resnet18v2/test_data_set_{i}")
    xs.append(x)
    ys.append(y)

x = np.concatenate(xs, axis=0)
y = np.array(ys)

with gzip.open("testdata.pkl.gz", "wb") as f:
    pickle.dump((x, y, NCAT), f)
