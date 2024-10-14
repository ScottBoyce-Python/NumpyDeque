import pytest
import numpy as np
from NumpyDeque import NumpyDeque


@pytest.fixture
def empty_deque():
    return NumpyDeque(maxsize=5)


@pytest.fixture
def filled_deque():
    return NumpyDeque.array([1, 2, 3, 4, 5], maxsize=5)


def test_edge_case_pop_empty(empty_deque):
    assert empty_deque.pop() is None


def test_edge_case_popleft_empty(empty_deque):
    assert empty_deque.popleft() is None


def test_popleft(filled_deque):
    value = filled_deque.popleft()
    assert value == 1
    assert len(filled_deque) == 4
    assert (filled_deque.deque == [2, 3, 4, 5]).all()
    value = filled_deque.popleft()
    assert value == 2
    assert len(filled_deque) == 3
    assert (filled_deque.deque == [3, 4, 5]).all()


def test_pop(filled_deque):
    value = filled_deque.pop()
    assert value == 5
    assert len(filled_deque) == 4
    assert (filled_deque.deque == [1, 2, 3, 4]).all()

    value = filled_deque.pop()
    assert value == 4
    assert len(filled_deque) == 3
    assert (filled_deque.deque == [1, 2, 3]).all()


def test_pop_popleft(filled_deque):
    value = filled_deque.pop()
    assert value == 5
    assert len(filled_deque) == 4
    assert (filled_deque.deque == [1, 2, 3, 4]).all()

    value = filled_deque.popleft()
    assert value == 1
    assert len(filled_deque) == 3
    assert (filled_deque.deque == [2, 3, 4]).all()

    value = filled_deque.popleft()
    assert value == 2
    assert len(filled_deque) == 2
    assert (filled_deque.deque == [3, 4]).all()

    value = filled_deque.pop()
    assert value == 4
    assert len(filled_deque) == 1
    assert (filled_deque.deque == [3]).all()

    value = filled_deque.popleft()
    assert value == 3
    assert len(filled_deque) == 0
    assert (filled_deque.deque == []).all()

    value = filled_deque.popleft()
    assert value is None
    assert len(filled_deque) == 0
    assert (filled_deque.deque == []).all()


def test_drop_right_shift(filled_deque):
    value = filled_deque.drop(1)
    assert value == 2
    assert len(filled_deque) == 4
    assert (filled_deque.deque == [1, 3, 4, 5]).all()


def test_drop_left_shift(filled_deque):
    value = filled_deque.drop(3)
    assert value == 4
    assert len(filled_deque) == 4
    assert (filled_deque.deque == [1, 2, 3, 5]).all()


def test_drop_to_pop(filled_deque):
    value = filled_deque.drop(-1)
    assert value == 5
    assert len(filled_deque) == 4
    assert (filled_deque.deque == [1, 2, 3, 4]).all()


def test_drop_to_popleft(filled_deque):
    value = filled_deque.drop(0)
    assert value == 1
    assert len(filled_deque) == 4
    assert (filled_deque.deque == [2, 3, 4, 5]).all()


def test_repeated_drop(filled_deque):
    # [1, 2, 3, 4, 5]
    value = filled_deque.drop(2)
    assert value == 3
    value = filled_deque.drop(2)
    assert value == 4
    assert len(filled_deque) == 3
    value = filled_deque.drop(2)
    assert value == 5
    assert len(filled_deque) == 2
    value = filled_deque.drop(2)
    assert value == 2
    assert len(filled_deque) == 1
    value = filled_deque.drop(2)
    assert value == 1
    assert len(filled_deque) == 0
    assert (filled_deque.deque == []).all()


def test_repeated_drop_put(filled_deque):
    # [1, 2, 3, 4, 5]
    value = filled_deque.drop(2)  # [1, 2, 4, 5]
    filled_deque.put(33)  # [1, 2, 4, 5, 33]
    value = filled_deque.drop(2)  # [1, 2, 5, 33]
    assert value == 4
    filled_deque.putleft(44)
    assert (filled_deque.deque == [44, 1, 2, 5, 33]).all()
