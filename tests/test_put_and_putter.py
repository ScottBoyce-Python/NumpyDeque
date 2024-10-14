import pytest
import numpy as np
from NumpyDeque import NumpyDeque


@pytest.fixture
def empty_deque():
    return NumpyDeque(maxsize=5)


def test_put(empty_deque):
    empty_deque.put(1)
    assert len(empty_deque) == 1
    assert empty_deque.deque[-1] == 1

    empty_deque.put(2)
    assert len(empty_deque) == 2
    assert empty_deque.deque[-1] == 2


def test_putleft(empty_deque):
    empty_deque.putleft(1)
    assert len(empty_deque) == 1
    assert empty_deque.deque[0] == 1

    empty_deque.putleft(2)
    assert len(empty_deque) == 2
    assert empty_deque.deque[0] == 2


def test_put_over_maxsize():
    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)
    d.put(6)
    assert len(d) == 5
    assert (d.deque == [2, 3, 4, 5, 6]).all()


def test_putleft_over_maxsize():
    d = NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)
    d.putleft(6)
    assert len(d) == 5
    assert (d.deque == [6, 1, 2, 3, 4]).all()


def test_over_maxsize_putter(empty_deque):
    empty_deque.putter([1, 2, 3, 4, 5, 6, 7])
    assert len(empty_deque) == 5
    assert (empty_deque.deque == [3, 4, 5, 6, 7]).all()

    array_like = [10, 20, 30, 40, 50]
    d = NumpyDeque.array(array_like, _force_buffer_size=10)
    d.putter([1, 2, 3, 4, 5, 6, 7])
    assert (d.deque == [3, 4, 5, 6, 7]).all()

    array_like = [10, 20, 30, 40, 50]
    d = NumpyDeque.array(array_like, _force_buffer_size=10)
    d.putleft(0)
    d.putter([1, 2, 3, 4, 5, 6, 7])
    assert (d.deque == [3, 4, 5, 6, 7]).all()

    array_like = [10, 20, 30, 40, 50]
    d = NumpyDeque.array(array_like, _force_buffer_size=10)
    d.put(60)
    d.putter([1, 2, 3, 4, 5, 6, 7])
    assert (d.deque == [3, 4, 5, 6, 7]).all()


def test_over_maxsize_putter_buffer_shift():
    array_like = [1, 2, 3, 4, 5]
    d = NumpyDeque.array(array_like, priority="rightonly", _force_buffer_size=10)
    assert d._s == 0  # buffer shift index
    d.put(6)
    d.put(7)
    d.putter([8, 9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([7, 8, 9, 10, 11])).all()

    d = NumpyDeque.array(array_like, priority="right", _force_buffer_size=10)
    assert d._s == 1  # buffer shift index
    d.put(6)
    d.put(7)
    d.putter([8, 9, 10])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([6, 7, 8, 9, 10])).all()


def test_over_maxsize_putterleft(empty_deque):
    empty_deque.putterleft([1, 2, 3, 4, 5, 6, 7])
    assert len(empty_deque) == 5
    assert (empty_deque.deque == [7, 6, 5, 4, 3]).all()

    array_like = [10, 20, 30, 40, 50]
    d = NumpyDeque.array(array_like, _force_buffer_size=10)
    d.putterleft([1, 2, 3, 4, 5, 6, 7])
    assert (d.deque == [7, 6, 5, 4, 3]).all()

    array_like = [10, 20, 30, 40, 50]
    d = NumpyDeque.array(array_like, _force_buffer_size=10)
    d.putleft(0)
    d.putterleft([1, 2, 3, 4, 5, 6, 7])
    assert (d.deque == [7, 6, 5, 4, 3]).all()

    array_like = [10, 20, 30, 40, 50]
    d = NumpyDeque.array(array_like, _force_buffer_size=10)
    d.put(60)
    d.putterleft([1, 2, 3, 4, 5, 6, 7])
    assert (d.deque == [7, 6, 5, 4, 3]).all()


def test_over_maxsize_putterleft_buffer_shift1():
    array_like = [1, 2, 3, 4, 5]
    d = NumpyDeque.array(array_like, priority="rightonly", _force_buffer_size=10)
    assert d._s == 0  # buffer shift index
    d.put(6)  # [2, 3, 4, 5, 6]
    d.put(7)  # [3, 4, 5, 6, 7]
    d.putterleft([8, 9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([11, 10, 9, 8, 3])).all()

    d = NumpyDeque.array(array_like, priority="rightonly", _force_buffer_size=10)
    assert d._s == 0  # buffer shift index
    d.put(6)  # [2, 3, 4, 5, 6]
    d.putterleft([8, 9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([11, 10, 9, 8, 2])).all()

    d = NumpyDeque.array(array_like, priority="rightonly", _force_buffer_size=10)
    assert d._s == 0  # buffer shift index
    d.put(6)  # [2, 3, 4, 5, 6]
    d.putterleft([8, 9, 10])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([10, 9, 8, 2, 3])).all()

    d = NumpyDeque.array(array_like, priority="rightonly", _force_buffer_size=10)
    assert d._s == 0  # buffer shift index
    d.putleft(0)  # [0, 1, 2, 3, 4]
    d.putterleft([8, 9, 10])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([10, 9, 8, 0, 1])).all()

    d = NumpyDeque.array(array_like, priority="rightonly", _force_buffer_size=10)
    assert d._s == 0  # buffer shift index
    d.putleft(0)  # [0, 1, 2, 3, 4]
    d.putleft(-1)  # [-1, 0, 1, 2, 3, 4]
    d.putterleft([9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([11, 10, 9, -1, 0])).all()


def test_over_maxsize_putterleft_buffer_shift2():
    array_like = [1, 2, 3, 4, 5]
    d = NumpyDeque.array(array_like, priority="right", _force_buffer_size=10)
    assert d._s == 1  # buffer shift index
    d.put(6)
    d.put(7)
    d.putterleft([8, 9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([11, 10, 9, 8, 3])).all()

    d = NumpyDeque.array(array_like, priority="right", _force_buffer_size=10)
    assert d._s == 1  # buffer shift index
    d.put(6)
    d.putterleft([8, 9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([11, 10, 9, 8, 2])).all()

    d = NumpyDeque.array(array_like, priority="right", _force_buffer_size=10)
    assert d._s == 1  # buffer shift index
    d.putleft(0)
    d.putleft(-1)
    d.putterleft([8, 9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([11, 10, 9, 8, -1])).all()

    d = NumpyDeque.array(array_like, priority="right", _force_buffer_size=10)
    assert d._s == 1  # buffer shift index
    d.putleft(0)
    d.putterleft([8, 9, 10, 11])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([11, 10, 9, 8, 0])).all()

    d = NumpyDeque.array(array_like, priority="right", _force_buffer_size=10)
    assert d._s == 1  # buffer shift index
    d.putleft(0)  # [0, 1, 2, 3, 4]
    d.putterleft([8, 9, 10])  # shift occurs
    assert d._i == d._s
    assert (d.deque == np.array([10, 9, 8, 0, 1])).all()
