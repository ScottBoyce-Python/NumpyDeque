from NumpyDeque import NumpyDeque
import numpy as np


def test_initialization_array():
    array_like = [1, 2, 3, 4, 5]
    d = NumpyDeque.array(array_like)
    assert len(d) == 5
    assert d.maxsize == 5
    assert np.all(d.deque == np.array(array_like))

    d = NumpyDeque.array(array_like, maxsize=10)
    assert len(d) == 5
    assert d.maxsize == 10
    assert np.all(d.deque == np.array(array_like))


def test_empty_initialization():
    d = NumpyDeque(maxsize=10)
    assert len(d) == 0
    assert d.maxsize == 10
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=100)
    assert len(d) == 0
    assert d.maxsize == 100
    assert len(d.deque) == 0


def test_fill_initialization():
    d = NumpyDeque(maxsize=10, fill=1)
    assert len(d) == 10
    assert d.maxsize == 10
    assert np.all(d.deque == 1)
    assert len(d.deque) == 10

    d = NumpyDeque(maxsize=100, fill=2)
    assert len(d) == 100
    assert d.maxsize == 100
    assert np.all(d.deque == 2)
    assert len(d.deque) == 100


def test_int64_initialization():
    d = NumpyDeque(maxsize=10, dtype=np.int64)
    assert d.dtype == np.int64
    assert len(d) == 0
    assert d.maxsize == 10
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=100, dtype=np.int64)
    assert d.dtype == np.int64
    assert len(d) == 0
    assert d.maxsize == 100
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=10, fill=1, dtype=np.int64)
    assert d.dtype == np.int64
    assert len(d) == 10
    assert d.maxsize == 10
    assert np.all(d.deque == 1)
    assert len(d.deque) == 10

    d = NumpyDeque(maxsize=100, fill=2, dtype=np.int64)
    assert d.dtype == np.int64
    assert len(d) == 100
    assert d.maxsize == 100
    assert np.all(d.deque == 2)
    assert len(d.deque) == 100


def test_float64_initialization():
    d = NumpyDeque(maxsize=10, dtype=np.float64)
    assert d.dtype == np.float64
    assert len(d) == 0
    assert d.maxsize == 10
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=100, dtype=np.float64)
    assert d.dtype == np.float64
    assert len(d) == 0
    assert d.maxsize == 100
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=10, fill=1, dtype=np.float64)
    assert d.dtype == np.float64
    assert len(d) == 10
    assert d.maxsize == 10
    assert np.all(d.deque == 1.0)
    assert len(d.deque) == 10

    d = NumpyDeque(maxsize=100, fill=2, dtype=np.float64)
    assert d.dtype == np.float64
    assert len(d) == 100
    assert d.maxsize == 100
    assert np.all(d.deque == 2.0)
    assert len(d.deque) == 100


def test_pad_initialization():
    siz = 10
    pad = 10
    d = NumpyDeque(maxsize=siz, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == 0
    assert d.maxsize == siz

    siz = 100
    pad = 10
    d = NumpyDeque(maxsize=siz, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == 0
    assert d.maxsize == siz

    siz = 10
    pad = 100
    d = NumpyDeque(maxsize=siz, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == 0
    assert d.maxsize == siz

    siz = 100
    pad = 100
    d = NumpyDeque(maxsize=siz, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == 0
    assert d.maxsize == siz

    siz = 10
    pad = 10
    d = NumpyDeque(maxsize=siz, fill=1, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == siz
    assert d.maxsize == siz
    assert np.all(d.deque == 1)

    siz = 100
    pad = 10
    d = NumpyDeque(maxsize=siz, fill=2, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == siz
    assert d.maxsize == siz
    assert np.all(d.deque == 2)

    siz = 10
    pad = 100
    d = NumpyDeque(maxsize=siz, fill=1, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == siz
    assert d.maxsize == siz
    assert np.all(d.deque == 1)

    siz = 100
    pad = 100
    d = NumpyDeque(maxsize=siz, fill=2, buffer_padding=pad)
    assert d.buffer_size >= siz + 2 * pad
    assert len(d) == siz
    assert d.maxsize == siz
    assert np.all(d.deque == 2)


def test_priority_initialization():
    d = NumpyDeque(maxsize=10, priority="right")
    assert len(d) == 0
    assert d.maxsize == 10
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=10, fill=1, priority="right")
    assert len(d) == 10
    assert d.maxsize == 10
    assert np.all(d.deque == 1)
    assert len(d.deque) == 10

    d = NumpyDeque(maxsize=100, priority="right")
    assert len(d) == 0
    assert d.maxsize == 100
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=100, fill=1, priority="right")
    assert len(d) == 100
    assert d.maxsize == 100
    assert np.all(d.deque == 1)
    assert len(d.deque) == 100

    d = NumpyDeque(maxsize=10, priority="rightonly")
    assert len(d) == 0
    assert d.maxsize == 10
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=10, fill=1, priority="rightonly")
    assert len(d) == 10
    assert d.maxsize == 10
    assert np.all(d.deque == 1)
    assert len(d.deque) == 10

    d = NumpyDeque(maxsize=100, priority="rightonly")
    assert len(d) == 0
    assert d.maxsize == 100
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=100, fill=1, priority="rightonly")
    assert len(d) == 100
    assert d.maxsize == 100
    assert np.all(d.deque == 1)
    assert len(d.deque) == 100

    d = NumpyDeque(maxsize=10, priority="left")
    assert len(d) == 0
    assert d.maxsize == 10
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=10, fill=1, priority="left")
    assert len(d) == 10
    assert d.maxsize == 10
    assert np.all(d.deque == 1)
    assert len(d.deque) == 10

    d = NumpyDeque(maxsize=100, priority="left")
    assert len(d) == 0
    assert d.maxsize == 100
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=100, fill=1, priority="left")
    assert len(d) == 100
    assert d.maxsize == 100
    assert np.all(d.deque == 1)
    assert len(d.deque) == 100

    d = NumpyDeque(maxsize=10, priority="leftonly")
    assert len(d) == 0
    assert d.maxsize == 10
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=10, fill=1, priority="leftonly")
    assert len(d) == 10
    assert d.maxsize == 10
    assert np.all(d.deque == 1)
    assert len(d.deque) == 10

    d = NumpyDeque(maxsize=100, priority="leftonly")
    assert len(d) == 0
    assert d.maxsize == 100
    assert len(d.deque) == 0

    d = NumpyDeque(maxsize=100, fill=1, priority="leftonly")
    assert len(d) == 100
    assert d.maxsize == 100
    assert np.all(d.deque == 1)
    assert len(d.deque) == 100
