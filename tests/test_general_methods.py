import pytest
import numpy as np
from NumpyDeque import NumpyDeque


def test_custom_dtype():
    dtype = np.dtype([("field1", np.int32), ("field2", np.float64)])
    d = NumpyDeque(maxsize=5, dtype=dtype)
    d.put((1, 2.5))
    d.put((3, 4.5))
    assert d[0]["field1"] == 1
    assert d[1]["field2"] == 4.5


def test_buffer_shift():
    array_like = [1, 2, 3, 4, 5]
    d = NumpyDeque.array(array_like, priority="rightonly", _force_buffer_size=10)
    assert d._s == 0  # buffer shift index
    assert len(d) == 5
    assert d.buffer_size == 10
    assert d.maxsize == 5
    for i in range(6, 11):
        d.put(i)
    d.put(11)  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([7, 8, 9, 10, 11])).all()
    for i in range(6):
        d.putleft(i)  # shift occurs always
    assert d._i == d._s
    assert (d.deque == np.array([5, 4, 3, 2, 1])).all()

    array_like = [1, 2, 3, 4, 5]
    d = NumpyDeque.array(array_like, priority="right", _force_buffer_size=10)
    assert d._s == 1  # buffer shift index
    assert len(d) == 5
    assert d.buffer_size == 10
    assert d.maxsize == 5
    for i in range(6, 10):
        d.put(i)
    d.put(10)  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([6, 7, 8, 9, 10])).all()
    d.putleft(0)
    d.putleft(1)  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([1, 0, 6, 7, 8])).all()

    d = NumpyDeque.array(array_like, priority="left", _force_buffer_size=10)
    assert d._s == 3  # buffer shift index
    assert len(d) == 5
    assert d.buffer_size == 10
    assert d.maxsize == 5
    for i in range(6, 8):
        d.put(i)  # shift occurs
    d.put(8)  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([4, 5, 6, 7, 8])).all()
    for i in range(3):
        d.putleft(i)
    d.putleft(3)  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([3, 2, 1, 0, 4])).all()

    d = NumpyDeque.array(array_like, priority="leftonly", _force_buffer_size=10)
    assert d._s == 5  # buffer shift index
    assert len(d) == 5
    assert d.buffer_size == 10
    assert d.maxsize == 5
    for i in range(6, 11):
        d.put(i)  # shift occurs always
    d.put(11)  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([7, 8, 9, 10, 11])).all()
    for i in range(5):
        d.putleft(i)
    d.putleft(5)  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([5, 4, 3, 2, 1])).all()


def test_invalid_priority():
    with pytest.raises(ValueError):
        NumpyDeque(maxsize=5, priority="invalid")


def test_clear():
    d = NumpyDeque.array([1, 2, 3, 4, 5])
    d.clear()
    assert len(d) == 0


def test_copy():
    d = NumpyDeque.array([1, 2, 3, 4, 5])
    copied_deque = d.copy()
    assert len(copied_deque) == len(d)
    assert (copied_deque.deque == d.deque).all()
    assert copied_deque is not d


def test_reverse():
    d = NumpyDeque.array([1, 2, 3, 4, 5])
    d.reverse()
    assert (d.deque == [5, 4, 3, 2, 1]).all()


def test_abs():
    d = NumpyDeque.array([-1, -2, 3, -4, 5], maxsize=5)
    d.abs()
    assert (d.deque == [1, 2, 3, 4, 5]).all()


def test_sort():
    d = NumpyDeque.array([3, 1, 4, 2, 5], maxsize=5)
    d.sort()
    assert (d.deque == [1, 2, 3, 4, 5]).all()
