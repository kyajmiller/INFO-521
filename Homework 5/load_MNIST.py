import numpy as np
import sys


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)


def load_MNIST_images(filename):
    """
    returns a 28x28x[number of MNIST images] matrix containing
    the raw MNIST images
    :param filename: input data file
    """
    with open(filename, "rb") as f:
        magic = _read32(f)

        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))

        num_images = _read32(f)
        num_rows = _read32(f)
        num_cols = _read32(f)

        images = np.fromfile(f, dtype=np.ubyte)

        images = images.reshape((num_images, num_rows * num_cols)).transpose()
        images = images.astype(np.float64) / 255

        f.close()

    return images


def load_MNIST_labels(filename):
    """
    returns a [number of MNIST images]x1 matrix containing
    the labels for the MNIST images

    :param filename: input file with labels
    """
    with open(filename, 'rb') as f:
        labels = np.fromfile(f, dtype=np.ubyte)

        f.close()

    return labels
