# NumpyDeque Python Module

<p align="left">
  <img src="https://github.com/ScottBoyce-Python/NumpyDeque/actions/workflows/python-pytest.yml/badge.svg" alt="Build Status" height="20">
</p>



NumpyDeque is a `numpy.ndarray` based double-ended queue (`deque`) with a maximum size. The deque has an initial size of zero and grows as values are added. When the deque has `maxsize` values and another value is added (`put` or `putleft`), the value on the opposite end is dropped.  The underlying buffer allows fast addition and removal of values from the deque on both ends. 

This double-ended queue is efficiently done by using a padded-buffer array. When an operation results no more padding the buffer performs an internal shift to restore the padding buffer space. The padding can give priority to specific operations, where more padding is given for faster `put` operations (adding to the end), or for `putleft` operations (adding to start).

## Features

- The deque has most of the `collections.deque` methods.
- The deque can use any `nump.ndarray` methods..
- Supports fast append and pop operations on both ends of the deque.

## Installation
Ensure that `numpy` is installed in your environment. If not, you can install it using:  
 (note, this module was only tested against `numpy>2.0`)

```bash
pip install numpy
```

To install the module

```bash
pip install --upgrade git+https://github.com/ScottBoyce-Python/NumpyDeque.git
```

or you can clone the respository with
```bash
git clone https://github.com/ScottBoyce-Python/NumpyDeque.git
```
and then move the file `NumpyDeque/NumpyDeque.py` to wherever you want to use it.


## Usage

Below are examples showcasing how to create and interact with a `NumpyDeque`.

### Creating a Deque

```python
import numpy as np
from NumpyDeque import NumpyDeque

# Create an empty deque that stores up to 10 float64 numbers
d = NumpyDeque(maxsize=10, dtype=np.float64)

# Create a deque with 5 int64 zeros (the deque is initialized to maxsize with 0).
d = NumpyDeque(maxsize=5, fill=0, dtype=np.int64)

# Create a deque from an array. Its maxsize is automatically set to 5.
d = NumpyDeque.array([1, 2, 3, 4, 5])

# Create a deque from an array. Its maxsize is set to 5.
d = NumpyDeque.array([1, 2, 3], 5)

```

### Adding to Right of The Deque

```python
d = NumpyDeque(maxsize=5, dtype=np.int64)

# Put a value to the right on the deque
d.put(5)
d.put(7)
d.put(9)
print(d)              # Output: NumpyDeque([5, 7, 9])
d.put(11)
d.put(13)
print(d)              # Output: NumpyDeque([5, 7, 9, 11, 13])
d.put(15)  # 5 is dropped
print(d)              # Output: NumpyDeque([7, 9, 11, 13, 15])

d.putter([1, 2, 3])
print(d)              # Output: NumpyDeque([13, 15, 1, 2, 3])

d.putter([-1, -2, -3, -4, -5, -6, -7])
print(d)              # Output: NumpyDeque([-3, -4, -5, -6, -7])

d.putter([1, 2, 3, 4, 5])
print(d)              # Output: NumpyDeque([1, 2, 3, 4, 5])
```

### Adding to Left of The Deque

```python
d = NumpyDeque(maxsize=5, dtype=np.int64)

# Put a value to the right on the deque
d.putleft(5)
d.putleft(7)
d.putleft(9)
print(d)              # Output: NumpyDeque([9, 7, 5])
d.putleft(11)
d.putleft(13)
print(d)              # Output: NumpyDeque([13, 11, 9, 7, 5])
d.putleft(15)  # 5 is dropped
print(d)              # Output: NumpyDeque([15, 13, 11, 9, 7])

d.putterleft([1, 2, 3])
print(d)              # Output: NumpyDeque([3, 2, 1, 15, 13])

d.putterleft([-1, -2, -3, -4, -5, -6, -7])
print(d)              # Output: NumpyDeque([-7, -6, -5, -4, -3])

d.putter([1, 2, 3, 4, 5])
print(d)              # Output: NumpyDeque([5, 4, 3, 2, 1])
```

### Removing Elements

```python
d = NumpyDeque.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Remove and return the last element
print(d)              # Output: NumpyDeque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
rightmost_value = d.pop() 
print(d)              # Output: NumpyDeque([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(rightmost_value)# Output: 10

# Remove and return the first element
leftmost_value = d.popleft() 
print(d)              # Output: NumpyDeque([2, 3, 4, 5, 6, 7, 8, 9])
print(leftmost_value) # Output: 1

# Remove and return the third element
third_value = d.drop(2) 
print(d)              # Output: NumpyDeque([2, 3, 5, 6, 7, 8, 9])
print(third_value)# Output: 4

# If the number 8 and 1 are found, remove the first appearance
d.remove(8)
print(d)              # Output: NumpyDeque([2, 3, 5, 6, 7, 9])
d.remove(1)           # Nothing happens
print(d)              # Output: NumpyDeque([2, 3, 5, 6, 7, 9])
```

### Slicing

```python
d = NumpyDeque.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], maxsize=10)

# Slice behaves like NumPy arrays, but be careful with indexes
print( d[1:4] )    # Output: [1, 2, 3]

d[1:3] = [-1, -2]     # Output: [2, 3, 4]
print(d)              # Output: NumpyDeque([0, -1, -2, 3, 4, 5, 6, 7, 8, 9])

# Note that values move once maxsize is exceeded
print( d[2] )         # Output: -2
d.put(10)
print(d)              # Output: NumpyDeque([-1, -2, 3, 4, 5, 6, 7, 8, 9, 10])
print( d[2] )         # Output: 3
d.put(11)
print(d)              # Output: NumpyDeque([-2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
print( d[2] )         # Output: 4
d.putleft(99)
print(d)              # Output: NumpyDeque([99, -2, 3, 4, 5, 6, 7, 8, 9, 10])
print( d[2] )         # Output: 3

#Be careful about the size
d = NumpyDeque(maxsize=5)
d.put(5)
d.put(4)
d.put(3)
print(d)              # Output: NumpyDeque([5, 4, 3])
print( d[3] )         # Raises index error!!!
```

## Testing

This project uses `pytest` and `pytest-xdist` for testing. Tests are located in the `tests` folder. To run tests, install the required packages and execute the following command:

```bash
pip install pytest pytest-xdist

pytest  # run all tests, note options are set in the pyproject.toml file
```

&nbsp; 

Note, that the [pyproject.toml](pyproject.toml) contains the flags used for pytest.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Author
Scott E. Boyce
