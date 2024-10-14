import pytest
import numpy as np
from NumpyDeque import NumpyDeque


@pytest.fixture
def filled_deque():
    return NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)


def test_pop_index(filled_deque):
    assert filled_deque[2] == 3
    filled_deque.pop()
    assert filled_deque[2] == 3
    filled_deque.pop()
    assert filled_deque[2] == 3
    filled_deque.pop()
    with pytest.raises(IndexError):
        filled_deque[2]
    assert (filled_deque == [1, 2]).all()


def test_popleft_index(filled_deque):
    assert filled_deque[2] == 3
    filled_deque.popleft()
    assert filled_deque[2] == 4
    filled_deque.popleft()
    assert filled_deque[2] == 5
    filled_deque.popleft()
    with pytest.raises(IndexError):
        filled_deque[2]
    assert (filled_deque == [4, 5]).all()


def test_slicing_boundaries():
    d = NumpyDeque(maxsize=5, dtype=np.int64)
    d.put(1)
    d.put(2)
    assert (d[0:10] == [1, 2]).all()  # Slice beyond the current size should return available elements
    assert (d[-5:] == [1, 2]).all()  # Negative slicing


def test_equality():
    d1 = NumpyDeque(maxsize=5, dtype=np.int64)
    d2 = NumpyDeque(maxsize=5, dtype=np.int64)
    d1.put(1)
    d2.put(1)
    assert (d1 == d2).all()
    d2.put(2)
    assert (d1 != d2).any()


def test_inplace_add_operations():
    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)
    d += 10
    assert len(d) == 5
    assert (d == [11, 12, 13, 14, 15]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d += 10
    assert (d == [11, 12, 13, 14, 15]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d.put(6)
    d += 10
    assert (d == [11, 12, 13, 14, 15, 16]).all()


def test_inplace_subtract_operations():
    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)
    d -= 10
    assert len(d) == 5
    assert (d == [-9, -8, -7, -6, -5]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d -= 10
    assert (d == [-9, -8, -7, -6, -5]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d.put(6)
    d -= 10
    assert (d == [-9, -8, -7, -6, -5, -4]).all()


def test_inplace_multiply_operations():
    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)
    d *= 10
    assert len(d) == 5
    assert (d == [10, 20, 30, 40, 50]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d *= 10
    assert (d == [10, 20, 30, 40, 50]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d.put(6)
    d *= 10
    assert (d == [10, 20, 30, 40, 50, 60]).all()


def test_inplace_divide_operations():
    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5, dtype=float)
    d /= 10
    assert (np.isclose(d, [0.1, 0.2, 0.3, 0.4, 0.5])).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5, dtype=int)
    with pytest.raises(Exception):
        d /= 10

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d //= 10
    assert (d == [0, 0, 0, 0, 0]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d.put(6)
    d //= 10
    assert (d == [0, 0, 0, 0, 0, 0]).all()


def test_inplace_power_operations():
    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5, dtype=float)
    d **= 2.5
    assert (np.isclose(d, [1.0, 5.656, 15.588, 32.0, 55.901], atol=0.001)).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5, dtype=int)
    with pytest.raises(Exception):
        d **= 2.5

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d **= 2
    assert (d == [1, 4, 9, 16, 25]).all()

    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=10)
    d.put(6)
    d **= 2
    assert (d == [1, 4, 9, 16, 25, 36]).all()
