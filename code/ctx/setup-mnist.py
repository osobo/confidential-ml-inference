import gzip
import numpy as np
import pickle

NCAT = 10

def file2np(filename, skip, shape):
    with gzip.open(filename, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=skip).reshape(shape)

x = file2np("t10k-images-idx3-ubyte.gz", 16, (-1, 1, 1, 28, 28))
y = file2np("t10k-labels-idx1-ubyte.gz", 8, (-1,))

x = x.astype(np.float32) / 255

with gzip.open("testdata.pkl.gz", "wb") as f:
    pickle.dump((x, y, NCAT), f)
