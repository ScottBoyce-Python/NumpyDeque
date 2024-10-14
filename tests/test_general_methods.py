import pytest
import numpy as np
from NumpyDeque import NumpyDeque


@pytest.fixture
def empty_deque():
    return NumpyDeque(maxsize=5)


@pytest.fixture
def filled_deque():
    return NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)


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
    assert np.all(d.deque == np.array([7, 8, 9, 10, 11]))
    for i in range(6):
        d.putleft(i)  # shift occurs always
    assert d._i == d._s
    assert np.all(d.deque == np.array([5, 4, 3, 2, 1]))

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
    assert np.all(d.deque == np.array([6, 7, 8, 9, 10]))
    d.putleft(0)
    d.putleft(1)  # shift occurs
    assert d._i == d._s
    assert np.all(d.deque == np.array([1, 0, 6, 7, 8]))

    d = NumpyDeque.array(array_like, priority="left", _force_buffer_size=10)
    assert d._s == 3  # buffer shift index
    assert len(d) == 5
    assert d.buffer_size == 10
    assert d.maxsize == 5
    for i in range(6, 8):
        d.put(i)  # shift occurs
    d.put(8)  # shift occurs
    assert d._i == d._s
    assert np.all(d.deque == np.array([4, 5, 6, 7, 8]))
    for i in range(3):
        d.putleft(i)
    d.putleft(3)  # shift occurs
    assert d._i == d._s
    assert np.all(d.deque == np.array([3, 2, 1, 0, 4]))

    d = NumpyDeque.array(array_like, priority="leftonly", _force_buffer_size=10)
    assert d._s == 5  # buffer shift index
    assert len(d) == 5
    assert d.buffer_size == 10
    assert d.maxsize == 5
    for i in range(6, 11):
        d.put(i)  # shift occurs always
    d.put(11)  # shift occurs
    assert d._i == d._s
    assert np.all(d.deque == np.array([7, 8, 9, 10, 11]))
    for i in range(5):
        d.putleft(i)
    d.putleft(5)  # shift occurs
    assert d._i == d._s
    assert np.all(d.deque == np.array([5, 4, 3, 2, 1]))


def test_invalid_priority():
    with pytest.raises(ValueError):
        NumpyDeque(maxsize=5, priority="invalid")


def test_clear(filled_deque):
    filled_deque.clear()
    assert len(filled_deque) == 0


def test_copy(filled_deque):
    copied_deque = filled_deque.copy()
    assert len(copied_deque) == len(filled_deque)
    assert np.array_equal(copied_deque.deque, filled_deque.deque)
    assert copied_deque is not filled_deque


def test_reverse(filled_deque):
    filled_deque.reverse()
    assert np.array_equal(filled_deque.deque, [5, 4, 3, 2, 1])


def test_abs():
    d = NumpyDeque.array([-1, -2, 3, -4, 5], maxsize=5)
    d.abs()
    assert np.array_equal(d.deque, [1, 2, 3, 4, 5])


def test_sort():
    d = NumpyDeque.array([3, 1, 4, 2, 5], maxsize=5)
    d.sort()
    assert np.array_equal(d.deque, [1, 2, 3, 4, 5])
