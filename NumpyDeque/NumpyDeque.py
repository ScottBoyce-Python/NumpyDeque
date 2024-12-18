import numpy as np

try:
    from ._metadata import (
        __version__,
        __author__,
        __email__,
        __license__,
        __status__,
        __maintainer__,
        __credits__,
        __url__,
        __description__,
        __copyright__,
    )
except ImportError:
    try:
        from _metadata import (
            __version__,
            __author__,
            __email__,
            __license__,
            __status__,
            __maintainer__,
            __credits__,
            __url__,
            __description__,
            __copyright__,
        )
    except ImportError:
        # _metadata.py failed to load,
        # fill in with dummy values (script may be standalone)
        __version__ = "Failed to load from _metadata.py"
        __author__ = __version__
        __email__ = __version__
        __license__ = __version__
        __status__ = __version__
        __maintainer__ = __version__
        __credits__ = __version__
        __url__ = __version__
        __description__ = __version__
        __copyright__ = __version__

# %% --------------------------------------------------------------------------

__all__ = [
    "NumpyDeque",
]

_PRIORITY_FLAG = {"e": "equal", "l": "left", "lo": "leftonly", "r": "right", "ro": "rightonly"}
_MAX_AUTO_BUFFER = 2**11  # Maximum size increase of buffer allowed to include user requested buffer and auto-buffer


class NumpyDeque:
    """
    A double-ended queue (deque) with a maximum size that is also a `numpy.ndarray` .
    The deque has an initial size of zero and grows as values are added.
    When the deque has maxsize values and another value is added (put or putleft),
    the value on the opposite end is dropped. The object's `deque` attribute
    is the numpy.ndarray double-ended queue.

    This double-ended queue is efficiently done by using a padded-buffer array.
    When an operation results no more padding the buffer performs an
    internal shift to restore the padding buffer space. The padding can give
    priority to specific operations, where more padding is given for faster
    `put` operations (adding to the end), or for `putleft` operations (adding to start).

    Attributes:
        deque (np.ndarray):     The current deque, representing the active elements.
        size (int):             Current number of elements in the deque.
        maxsize (int):          Maximum size of the deque. Once the limit is reached,
                                adding an element will discard the element on the opposite end.
        dtype (numpy.dtype):    Data type of the elements stored in the deque.
        priority (int):         The buffer priority for internal shifting of the deque.
                                Set to:
                                    "equal"    : buffer space gives equal priority to put and putleft
                                    "right"    : buffer space gives       priority to put
                                    "left"     : buffer space gives       priority to putleft
                                    "leftonly" : buffer space gives all   priority to put     (putleft -> internal shift)
                                    "rightonly": buffer space gives all   priority to putleft (put     -> internal shift)
        buffer_size (int):      The size of the underlying buffer that holds the deque.

    Private Attributes:
        _data (np.ndarray): The underlying NumPy array that stores the elements plus buffered space.
        _qcap (int): The maximum size of the deque, values added after this will remove a value.
        _bcap (int): The current capacity of the underlying np.ndarray buffer, `len(_data)`.
        _priority (str): Flag for how the operations are prioritized. Set to "r", "ro", "l", "lo", "e".
        _s (int): Starting index for shift operations within buffer when adding a value.
        _i (int): Index in _data that is the first value in the deque.
        _j (int): Index plus one in _data that is the last value in the deque.

    Constructors:
        array(array_like, maxsize=None): Create a NumpyDeque from an existing array-like object.

    Methods:
        put(value):           Adds an element to the right end of the deque.
                              If the deque is at maxsize, then removes the left most element.
        putleft(value):       Adds an element to the left end of the deque.
                              If the deque is at maxsize, then removes the right most element.

        putter(iterable):     Adds multiple elements to the right of the deque from an iterable.
                              If the deque is at maxsize, then removes the left most elements.
        putterleft(iterable): Adds multiple elements to the left of the deque from an iterable.
                              If the deque is at maxsize, then removes the right most elements.

        pop():                Removes and returns the rightmost element of the deque.
        popleft():            Removes and returns the leftmost element of the deque.
        remove(value):        Removes the first occurrence of the specified value from the deque.
        drop(index):          Removes and returns the element at the specified index.
        clear():              Removes all elements from the deque, leaving it empty.

        index(value):         Returns the index of the first occurrence of the specified value in the deque.
        count(value):         Returns the number of occurrences of the specified value in the deque.
        sort():               Sorts the order of the elements in the deque in-place.
        reverse():            Reverses the order of the elements in the deque in-place.


    Examples:
        >>> import numpy as np
        >>> from NumpyDeque import NumpyDeque

        >>> # Create an empty deque with a maximum size of 5
        >>> d = NumpyDeque(maxsize=5, dtype=np.int64)
        >>> d.put(1)
        >>> d.put(2)
        >>> print(d)
        NumpyDeque([1, 2])

        >>> # Add elements until it reaches max size
        >>> d.put(3)
        >>> d.put(4)
        >>> d.put(5)
        >>> print(d)
        NumpyDeque([1, 2, 3, 4, 5])
        >>> # Adding values that exceed maxsize results in the opposite end being dropped
        >>> d.put(6)  # To to right, so drops the left
        >>> print(d)
        NumpyDeque([2, 3, 4, 5, 6])  # 1 is discarded

        >>> # Adding to the left end of the deque
        >>> d.putleft(0)  # To to left, so drops the right
        >>> print(d)
        NumpyDeque([0, 2, 3, 4, 5])  # 6 is discarded from the right end

        >>> # Removing elements
        >>> rightmost = d.pop()
        >>> print(rightmost)  # Output: 5
        >>> print(d)  # NumpyDeque([0, 2, 3, 4])

        >>> # Creating a deque from an array
        >>> arr = np.array([10, 20, 30, 40])
        >>> d = NumpyDeque.array(arr)
        >>> print(d)
        NumpyDeque([10, 20, 30, 40])

        >>> # Using numpy array methods
        >>> mean_value = d.mean()
        >>> print(mean_value)  # Output: 25.0

        # Slicing works similar to numpy arrays
        >>> print(d[1:3])
        NumpyDeque([20, 30])
    """

    deque: np.ndarray  # The pointer to the current deque.
    _data: np.ndarray  # The underlying NumPy array that stores the elements.
    _qcap: int  # The maximum size of the deque, values added after this will drop the first.
    _bcap: int  # The current capacity of the underlying np.ndarray buffer, `len(_data)`.
    _priority: str  # flag for how the operations are prioritized. Set to "r", "ro", "l", "lo", "e"
    _s: int  # Index that is used when shifting within buffer to accommodate adding a value.
    _i: int  # starting index in _data that is the first value in the deque
    _j: int  # one plus the ending index in _data that is the last value in the deque

    def __init__(
        self,
        maxsize,
        fill=None,
        dtype=None,
        buffer_padding=None,
        priority="right",
        *,
        _force_buffer_size=None,
    ):
        """
        Initialize the NumpyDeque with a given maximum size and optional initial value.

        Parameters:
            maxsize (int):                  The maximum number of elements that the deque can hold.
            fill (number, optional):        Value to fill to maxsize the deque with. If `dtype` is not provided,
                                            the data type is inferred from `fill`. Defaults to None.
            dtype (np.dtype, optional):     The data type (`numpy.dtype`) of the deque elements.
                                            If not provided, it defaults to `np.int32`.
                                            If `dtype` is not provided, and `fill` is, the fill type is be inferred.
            buffer_padding (int, optional): The approximate buffer padding on each end of the deque.
                                            It influences how the internal buffer array is allocated.
                                            That is, len(_data) > 2*buffer_padding + maxsize. Defaults to None.
            priority (str, optional):       Determines the priority of buffer padding usage. Defaults to "right".
                                            Set to:
                                              "equal"    : buffer space gives equal priority to put and putleft
                                              "right"    : buffer space gives       priority to put
                                              "left"     : buffer space gives       priority to putleft
                                              "leftonly" : buffer space gives all   priority to put     (putleft -> internal shift)
                                              "rightonly": buffer space gives all   priority to putleft (put     -> internal shift)
            _force_buffer_size (int, optional): For debugging use only. Overrides and sets the buffer (_data) size,
                                                such that it is equal to max(_force_buffer_size, maxsize). Default is None.
        """
        if maxsize < 1:
            maxsize = 0

        if fill is not None and dtype is None:
            try:
                dtype = fill.dtype
            except AttributeError:
                if type(fill) is int:
                    dtype = np.int32
                else:
                    dtype = np.array(fill).dtype

        if dtype is int or dtype is None:
            dtype = np.int32
        elif dtype is float:
            dtype = np.float64

        self._qcap = maxsize

        if _force_buffer_size is None:
            self._bcap = self._get_buffer_size(maxsize, buffer_padding)
        else:
            self._bcap = max(maxsize, _force_buffer_size)

        self._data = np.empty(self._bcap, dtype=dtype)

        priority = priority.strip().lower()
        p = priority[0]
        self._priority = p + "o" if "o" in priority else p

        pad = self._bcap - self._qcap
        left_pad = pad // 2
        # right_pad = pad - left_pad

        if pad < 2:
            self._s = 0
            self._priority = "ro"
        elif p == "e":  # equal padding on left and right
            self._s = left_pad
        elif p == "r":
            if "o" in priority:
                self._s = 0
            else:
                self._s = left_pad // 2
                if self._s == 0 and pad > 2:
                    self._s = 1
        elif p == "l":
            if "o" in priority:
                self._s = pad
            else:
                self._s = (3 * pad) // 4
                if self._s > pad:
                    self._s = pad
                if self._s == pad and pad > 2:
                    self._s = pad - 1
        else:
            raise ValueError(
                'NumpyDeque: `priority` must be set to "right", "left", "rightonly", "leftonly", or "equal"'
            )

        self._i = self._s

        if fill is None:
            self._j = self._i
        else:
            self._j = self._i + self._qcap
            self._data[self._i : self._j] = fill[0] if isinstance(fill, np.ndarray) else fill

        self.deque = self._data[self._i : self._j]

    @property
    def maxsize(self):
        """Get the maximum capacity of the deque."""
        return self._qcap

    @property
    def size(self):
        """Get the current number of elements in the deque."""
        return self._j - self._i

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the elements stored in the deque."""
        return self._data.dtype

    @property
    def priority(self):
        """Get the buffer priority of the deque."""
        return _PRIORITY_FLAG[self._priority]

    @property
    def buffer_size(self):
        """Get the maximum capacity of the buffer that contains the deque."""
        return self._bcap

    @classmethod
    def array(
        cls,
        object,
        maxsize=None,
        dtype=None,
        buffer_padding=None,
        priority="right",
        *,
        _force_buffer_size=None,
    ):
        """
        Create a NumpyDeque from an existing array-like object.

        This method allows initializing a NumpyDeque with values from an existing
        array-like object, using the length of that object as the maximum size.

        Parameters:
            object (array-like):            The array-like object whose values are used to initialize the deque.
            maxsize (int, optional):        The maximum number of elements that the deque can hold. Defaults to None.
                                            If None or less than len(object), then it is ignored.
            dtype (type, optional):         The data type for elements in the deque. If not provided, the type is inferred from
                                            the given object. Defaults to `np.array(object).dtype`.
            buffer_padding (int, optional): The approximate buffer padding on each end of the deque.
                                            It influences how the internal buffer array is allocated.
                                            That is, len(_data) > 2*buffer_padding + maxsize. Defaults to None.
            priority (str, optional):       Determines the priority of buffer padding usage. Defaults to "right".
                                            Set to:
                                              "equal"    : buffer space gives equal priority to put and putleft
                                              "right"    : buffer space gives       priority to put
                                              "left"     : buffer space gives       priority to putleft
                                              "leftonly" : buffer space gives all   priority to put     (putleft -> internal shift)
                                              "rightonly": buffer space gives all   priority to putleft (put     -> internal shift)
            _force_buffer_size (int, optional): For debugging use only. Overrides and sets the buffer (_data) size,
                                                such that it is equal to max(_force_buffer_size, maxsize). Default is None.

        Returns:
            NumpyDeque: A new instance of NumpyDeque initialized with values from the input object.

        Example:
            >>> NumpyDeque.array([1, 2, 3])
            NumpyDeque([1, 2, 3])
        """
        try:
            size = len(object)
        except TypeError:
            try:
                size = object.size
            except (AttributeError, TypeError):  # assume its a scalar
                if maxsize is not None:
                    maxsize = max(1, maxsize)
                return cls(maxsize, object, dtype, buffer_padding, _force_buffer_size)

        if maxsize is None:
            maxsize = size
        else:
            maxsize = max(size, maxsize)

        if dtype is None:
            try:
                dtype = object.dtype
            except AttributeError:
                if type(object[0]) is int:
                    dtype = np.int32
                else:
                    dtype = np.array(object[0]).dtype

        if buffer_padding is None and isinstance(object, NumpyDeque):
            _force_buffer_size = object._bcap

        ob = cls(maxsize, None, dtype, buffer_padding, priority, _force_buffer_size=_force_buffer_size)
        ob._i = ob._s
        ob._j = ob._i + size
        ob._data[ob._i : ob._j] = object
        ob.deque = ob._data[ob._i : ob._j]
        return ob

    def put(self, value):
        """
        Add a value to the right end of the deque. If the len(deque) == maxsize,
        the left-most value (index=0) is dropped.

        Parameters:
            value: The value to be added to the deque.
        """
        if self._j - self._i == self._qcap:
            self._i += 1

        if self._j == self._bcap:
            self._shift_buffer()

        self._data[self._j] = value
        self._j += 1
        self.deque = self._data[self._i : self._j]

    def putleft(self, value):
        """
        Add a value to the left end of the deque. If the len(deque) == maxsize,
        the right-most value (index=-1) is dropped.

        Parameters:
            value: The value to be added to the deque.
        """

        if self._j - self._i == self._qcap:
            self._j -= 1

        self._i -= 1
        if self._i == -1:
            self._shift_buffer()

        self._data[self._i] = value

        self.deque = self._data[self._i : self._j]

    def putter(self, values):
        """
        Add multiple values to the right of the deque in an optimized manner.
        This is more efficient than repeatedly calling `put()` for each value.
        This is equivalent to:
        ```
            for v in values:
                self.put(v)
        ```

        Parameters:
            values (iterable, array-like or scalar): The values to add to the deque.
        """
        try:
            dim = len(values)
        except TypeError:  # iterator so need to manually add values
            try:
                for v in values:
                    self.put(v)
            except TypeError:  # scalar, so just call regular put
                self.put(values)
            return

        if dim >= self._qcap:  # overwriting the entire deque move to buffer start
            self._i = self._s
            self._j = self._i + self._qcap
            self._data[self._i : self._j] = values[dim - self._qcap :]
            self.deque = self._data[self._i : self._j]
            return

        new_size = self._j - self._i + dim
        if new_size > self._qcap:
            self._i += new_size - self._qcap

        if self._j + dim >= self._bcap:
            self._shift_buffer()

        self._data[self._j : self._j + dim] = values
        self._j += dim
        self.deque = self._data[self._i : self._j]

    def putterleft(self, values):
        """
        Add multiple values to the left of the deque in an optimized manner.
        This is more efficient than repeatedly calling `putleft()` for each value.
        This is equivalent to:
        ```
            for v in values:
                self.putleft(v)
        ```

        Parameters:
            values (iterable, array-like or scalar): The values to add to the deque.
        """
        try:
            dim = len(values)
        except TypeError:  # iterator so need to manually add values
            try:
                for v in values:
                    self.putleft(v)
            except TypeError:  # scalar, so just call regular put
                self.putleft(values)
            return

        if dim >= self._qcap:  # overwriting the entire deque move to buffer start
            self._i = self._s
            self._j = self._i + self._qcap
            self._data[self._i : self._j] = values[::-1][: self._qcap]
            self.deque = self._data[self._i : self._j]
            return

        # if self._j - self._i == self._qcap:
        #     self._j -= 1

        # self._i -= 1
        # if self._i == -1:
        #     self._shift_buffer()

        # self._data[self._i] = value

        # self.deque = self._data[self._i : self._j]

        new_size = self._j - self._i + dim
        if new_size > self._qcap:
            self._j -= new_size - self._qcap

        if self._i - dim < 0:  # shift buffer
            s = self._s + dim
            j = s + self._j - self._i
            self._data[s:j] = self._data[self._i : self._j]
            self._i, self._j = self._s, j
        else:
            self._i -= dim

        self._data[self._i : self._i + dim] = values[::-1]
        self.deque = self._data[self._i : self._j]

    def pop(self):
        """
        Remove and return the value from the right end of the deque (index=-1).

        If the deque is empty, returns None.

        Returns:
            The value from the right end of the deque, or None if the deque is empty.
        """
        if self._i == self._j:
            return None
        self._j -= 1
        self.deque = self._data[self._i : self._j]
        return self._data[self._j]

    def popleft(self):
        """
        Remove and return the value from the left end of the deque (index=0).

        If the deque is empty, returns None.

        Returns:
            The value from the left end of the deque, or None if the deque is empty.
        """
        if self._i == self._j:
            return None
        self._i += 1
        self.deque = self._data[self._i : self._j]
        return self._data[self._i - 1]

    def drop(self, index):
        """
        Remove a value at the specified index from the deque, shifting subsequent elements.
        An invalid index results in pop or popleft, depending on which end it is beyond.

        Parameters:
            index (int): The index of the value to drop.
                         Supports negative indexing, where -1 is the last element.

        Returns:
            The value that was dropped.
        """
        # if index is less then the start of the deque, calls popleft
        # if index is greater than the end of the deque calls pop
        # otherwise drops the appropriate index and shifts the deque
        if index < 0:
            index = self._qcap + index

        index = self._i + index  # move deque index to _data index

        if index + 1 >= self._j:  # dropping last value
            return self.pop()

        if index <= self._i:  # dropping first value
            return self.popleft()

        value = self._data[index]  # value to return from popping

        if "o" in self._priority:  # user specified priority for buffer
            priority = self._priority
        elif (self._j - index) <= (index - self._i):  # index to drop is closer to _j
            priority = "ro"
        else:
            priority = "lo"

        if priority == "ro":  # have to shift left
            self._data[index : self._j - 1] = self._data[index + 1 : self._j]
            self._j -= 1
        else:  # priority == "lo" -> have to shift right
            self._data[self._i + 1 : index + 1] = self._data[self._i : index]
            self._i += 1

        self.deque = self._data[self._i : self._j]
        return value

    def sort(self):
        """
        Sort the elements within the deque in ascending order.

        Note:
            Sorting will change the order of elements,
            which affects the order in which elements are dropped.
        """
        self.deque.sort()

    def reverse(self):
        """
        Reverse the order of elements in the deque.

        Note:
            Reversing will change the order in which elements are dropped.
        """
        self._reverse_in_place(self.deque)

    def copy(self):
        """
        Create a new copy of the current deque with identical elements.

        Returns:
            NumpyDeque: A new instance of NumpyDeque that is a copy of the current deque.
        """
        return NumpyDeque.array(self.deque, None, self.dtype, None, self._priority, _force_buffer_size=self._bcap)

    def clear(self):
        """
        Clear all elements from the deque, resetting it to an empty state.
        """
        self._i = 0
        self._j = 0
        self.deque = self._data[self._i : self._j]

    def abs(self, where=True):
        """
        Compute the absolute values of all elements in the deque in place.

        Parameters:
            where (bool or array-like): A mask specifying which elements to update with their absolute values.
                                        Defaults to True, meaning all elements are updated.
        """
        np.absolute(self.deque, out=self.deque, where=where)

    def where(self, value):
        """
        Find and return the indices of all occurrences of the specified value in the deque.

        Parameters:
            value: The value to search for within the deque.

        Returns:
            np.ndarray: An array of indices where the specified value occurs in the deque.
        """
        return np.where(self.deque == value)[0]

    def index(self, value, start=0, stop=None):
        """
        Return the deque index of value (at or after index start and before index stop).
        Returns the first match

        Parameters:
            value: The value to search for.

        Returns:
            int: The the first index that value is found at.

        """
        p = self.where(value)
        if len(p) > 0:
            if stop is None:
                stop = self.size
            for i in p:
                if start <= i < stop:
                    return i
        return None

    def remove(self, value):
        """
        Remove the first occurrence of value.

        Parameters:
            value: The value to search for.
        """
        i = self.index(value)
        if i is not None:
            self.drop(i)

    def count(self, value):
        """
        Count the number of deque elements equal to value.

        Parameters:
            value: The value that is being counted.

        Returns:
            int: The count.
        """
        return np.count_nonzero(self.deque == value)

    def __getitem__(self, index):
        return self.deque[index]

    def __setitem__(self, index, value):
        self.deque[index] = value

    def __getattr__(self, name):
        """
        Handles unknown attributes by forwarding them to the NumPy.ndarray that holds the deque.
        This allows the NumpyDeque to support NumPy attributes such as `shape`, `dtype`, etc.

        Parameters:
            name (str): The name of the missing attribute.

        Returns:
            The value of the attribute from `self.deque`.

        Raises:
            AttributeError: If the attribute does not exist in `self` or `self.deque`.
        """

        try:
            # Forward any unknown attribute to the NumPy component
            attr = getattr(self.deque, name)
            if callable(attr):

                def method(*args, **kwargs):
                    return attr(*args, **kwargs)

                return method
            return attr
        except AttributeError:
            raise AttributeError(f"'NumpyDeque' object has no attribute '{name}'")

    def __len__(self):
        return self.size

    def __repr__(self):
        s = repr(self.deque)
        s = s[s.find("(") :]  # drop "array" from name
        return f"NumpyDeque{s}"

    def __str__(self):
        s = repr(self.deque)
        s = s[s.find("[") : s.find("]") + 1]  # get core part of numpy array
        return f"NumpyDeque({s})"

    def __iter__(self):
        return iter(self.deque)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, unused=None):
        return self.copy()

    def __contains__(self, value):
        return value in self.deque

    def __pos__(self):  # +val
        return self.copy()

    def __neg__(self):  # -val
        res = self.copy()
        res.deque *= -1
        return res

    def __abs__(self):  # abs(val)
        res = self.copy()
        res.abs()
        return res

    def __iadd__(self, other):  # To get called on addition with assignment e.g. a +=b.
        if isinstance(other, NumpyDeque):
            other = other.deque
        self.deque += other
        return self

    def __isub__(self, other):  # To get called on subtraction with assignment e.g. a -=b.
        if isinstance(other, NumpyDeque):
            other = other.deque
        self.deque -= other
        return self

    def __imul__(self, other):  # To get called on multiplication with assignment e.g. a *=b.
        if isinstance(other, NumpyDeque):
            other = other.deque
        self.deque *= other
        return self

    def __itruediv__(self, other):  # To get called on true division with assignment e.g. a /=b.
        if isinstance(other, NumpyDeque):
            other = other.deque
        self.deque /= other
        return self

    def __ifloordiv__(self, other):  # To get called on integer division with assignment e.g. a //=b.
        if isinstance(other, NumpyDeque):
            other = other.deque
        self.deque //= other
        return self

    def __imod__(self, other):  # To get called on modulo with assignment e.g. a%=b.
        if isinstance(other, NumpyDeque):
            other = other.deque
        self.deque %= other
        return self

    def __ipow__(self, other):  # To get called on exponents with assignment e.g. a **=b.
        if isinstance(other, NumpyDeque):
            other = other.deque
        self.deque **= other
        return self

    def __int__(self):  # To get called by built-int int() method to convert a type to an int.
        return NumpyDeque.array(self.deque, self.maxsize, np.int32, None, self.priority, _force_buffer_size=self._bcap)

    def __float__(self):  # To get called by built-int float() method to convert a type to float.
        return NumpyDeque.array(
            self.deque, self.maxsize, np.float64, None, self.priority, _force_buffer_size=self._bcap
        )

    def __add__(self, other):  # To get called on add operation using + operator
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):  # To get called on add operation using + operator
        res = self.copy()
        res += other
        return res

    def __sub__(self, other):  # To get called on subtraction operation using - operator.
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):  # To get called on subtraction operation using - operator.
        if isinstance(other, NumpyDeque):
            res = other.copy()
            res -= self.deque
        else:
            res = NumpyDeque.array(other)
            res -= self.deque
        return res

    def __mul__(self, other):  # To get called on multiplication operation using * operator.
        res = self.copy()
        res *= other
        return res

    def __rmul__(self, other):  # To get called on multiplication operation using * operator.
        res = self.copy()
        res *= other
        return res

    def __floordiv__(self, other):  # To get called on floor division operation using // operator.
        res = self.copy()
        res //= other
        return res

    def __rfloordiv__(self, other):  # To get called on floor division operation using // operator.
        if isinstance(other, NumpyDeque):
            res = other.copy()
            res //= self.deque
        else:
            res = NumpyDeque.array(other)
            res //= self.deque
        return res

    def __truediv__(self, other):  # To get called on division operation using / operator.
        res = self.copy()
        res /= other
        return res

    def __rtruediv__(self, other):  # To get called on division operation using / operator.
        if isinstance(other, NumpyDeque):
            res = other.copy()
            res /= self.deque
        else:
            res = NumpyDeque.array(other)
            res /= self.deque
        return res

    def __mod__(self, other):  # To get called on modulo operation using % operator.
        res = self.copy()
        res %= other
        return res

    def __rmod__(self, other):  # To get called on modulo operation using % operator.
        if isinstance(other, NumpyDeque):
            res = other.copy()
            res %= self.deque
        else:
            res = NumpyDeque.array(other)
            res %= self.deque
        return res

    def __pow__(self, other):  # To get called on calculating the power using ** operator.
        res = self.copy()
        res **= other
        return res

    def __rpow__(self, other):  # To get called on calculating the power using ** operator.
        if isinstance(other, NumpyDeque):
            res = other.copy()
            res **= self.deque
        else:
            res = NumpyDeque.array(other)
            res **= self.deque
        return res

    def __lt__(self, other):  # To get called on comparison using < operator.
        if isinstance(other, NumpyDeque):
            other = other.deque
        return self.deque < other

    def __le__(self, other):  # To get called on comparison using <= operator.
        if isinstance(other, NumpyDeque):
            other = other.deque
        return self.deque <= other

    def __gt__(self, other):  # To get called on comparison using > operator.
        if isinstance(other, NumpyDeque):
            other = other.deque
        return self.deque > other

    def __ge__(self, other):  # To get called on comparison using >= operator.
        if isinstance(other, NumpyDeque):
            other = other.deque
        return self.deque >= other

    def __eq__(self, other):  # To get called on comparison using == operator.
        if isinstance(other, NumpyDeque):
            other = other.deque
        return self.deque == other

    def __ne__(self, other):  # To get called on comparison using != operator.
        if isinstance(other, NumpyDeque):
            other = other.deque
        return self.deque != other

    def _shift_buffer(self):
        """
        Internally shift the buffer to accommodate newly added elements and ensure efficient use of buffer space.
        """
        if self._i == -1:  # putleft shift
            s = self._s + 1  # - self._i
            i = 0
            j = s + self._j  # - i
        elif self._i >= 0:  # put and putter shift
            s = self._s
            i = self._i
            j = self._s + self._j - self._i
        else:
            raise RuntimeError("NumpyDeque._shift_buffer encountered an invalid _i index.")

        self._data[s:j] = self._data[i : self._j]
        self._i, self._j = self._s, j

    @staticmethod
    def _get_buffer_size(size: int, buffer_padding: int) -> int:
        """
        Determine the appropriate buffer size based on the desired deque size
        and buffer_padding constraints.

        This method ensures that the buffer size is large enough to accommodate
        the deque and is close to a power of 2 for optimal memory usage.

        Parameters:
            size (int):           The size of the deque.
            buffer_padding (int): The minimum buffer padding on each end of the deque.

        Returns:
            int: The calculated total buffer size.
        """
        next2 = NumpyDeque._next_power_of_2
        if buffer_padding is None:
            if size < 4:
                return 32

            requested_buffer = 2 * size
            buffer_padding = next2(requested_buffer)
            if buffer_padding == requested_buffer:
                buffer_padding <<= 1
        else:
            requested_buffer = 2 * buffer_padding + size
            if size < 4 and requested_buffer < 8:
                return 8
            buffer_padding = next2(requested_buffer)

        while buffer_padding - requested_buffer < 8:
            buffer_padding <<= 1  # multiply by 2

        while buffer_padding - requested_buffer > _MAX_AUTO_BUFFER:
            buffer_padding -= 128

        return buffer_padding

    @staticmethod
    def _next_power_of_2(x: int):
        """
        Calculate the next power of 2 greater than or equal to x.

        Parameters:
            x (int): The input number.

        Returns:
            int: The next power of 2.
        """
        if x < 2:
            return 2
        x -= 1
        x |= x >> 1
        x |= x >> 2
        x |= x >> 4
        x |= x >> 8
        x |= x >> 16
        x |= x >> 32
        return x + 1

    @staticmethod
    def _reverse_in_place(array):
        """
        Reverse the contents of the given array in place.

        Parameters:
            array (np.ndarray): The array to reverse.
        """
        n = array.shape[0]
        for i in range(n // 2):
            array[i], array[n - i - 1] = array[n - i - 1], array[i]


if __name__ == "__main__":
    # Create an empty deque that stores up to 10 float64 numbers
    d = NumpyDeque(maxsize=10, dtype=np.float64)

    # Create a deque with 5 int64 zeros (the deque is initialized to maxsize with 0).
    d = NumpyDeque(maxsize=5, fill=0, dtype=np.int64)

    # Create a deque from an array. Its maxsize is automatically set to 5.
    d = NumpyDeque.array([1, 2, 3, 4, 5])

    # Create a deque from an array. Its maxsize is set to 5.
    d = NumpyDeque.array([1, 2, 3], 5)

    ### Adding to Right of The Deque

    d = NumpyDeque(maxsize=5, dtype=np.int64)

    # Put a value to the right on the deque
    d.put(5)
    d.put(7)
    d.put(9)
    print(d)  # Output: NumpyDeque([5, 7, 9])
    d.put(11)
    d.put(13)
    print(d)  # Output: NumpyDeque([5, 7, 9, 11, 13])
    d.put(15)  # 5 is dropped
    print(d)  # Output: NumpyDeque([7, 9, 11, 13, 15])

    d.putter([1, 2, 3])
    print(d)  # Output: NumpyDeque([13, 15, 1, 2, 3])

    d.putter([-1, -2, -3, -4, -5, -6, -7])
    print(d)  # Output: NumpyDeque([-3, -4, -5, -6, -7])

    d.putter([1, 2, 3, 4, 5])
    print(d)  # Output: NumpyDeque([1, 2, 3, 4, 5])

    ### Adding to Left of The Deque

    d = NumpyDeque(maxsize=5, dtype=np.int64)

    # Put a value to the right on the deque
    d.putleft(5)
    d.putleft(7)
    d.putleft(9)
    print(d)  # Output: NumpyDeque([9, 7, 5])
    d.putleft(11)
    d.putleft(13)
    print(d)  # Output: NumpyDeque([13, 11, 9, 7, 5])
    d.putleft(15)  # 5 is dropped
    print(d)  # Output: NumpyDeque([15, 13, 11, 9, 7])

    d.putterleft([1, 2, 3])
    print(d)  # Output: NumpyDeque([3, 2, 1, 15, 13])

    d.putterleft([-1, -2, -3, -4, -5, -6, -7])
    print(d)  # Output: NumpyDeque([-7, -6, -5, -4, -3])

    d.putterleft([1, 2, 3, 4, 5])
    print(d)  # Output: NumpyDeque([5, 4, 3, 2, 1])

    ### Removing Elements

    d = NumpyDeque.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Remove and return the last element
    print(d)  # Output: NumpyDeque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rightmost_value = d.pop()
    print(d)  # Output: NumpyDeque([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(rightmost_value)  # Output: 10

    # Remove and return the first element
    leftmost_value = d.popleft()
    print(d)  # Output: NumpyDeque([2, 3, 4, 5, 6, 7, 8, 9])
    print(leftmost_value)  # Output: 1

    # Remove and return the third element
    third_value = d.drop(2)
    print(d)  # Output: NumpyDeque([2, 3, 5, 6, 7, 8, 9])
    print(third_value)  # Output: 4

    # If the number 8 and 1 are found, remove the first appearance
    d.remove(8)
    print(d)  # Output: NumpyDeque([2, 3, 5, 6, 7, 9])
    d.remove(1)  # Nothing happens
    print(d)  # Output: NumpyDeque([2, 3, 5, 6, 7, 9])

    ### Slicing

    d = NumpyDeque.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], maxsize=10)

    # Slice behaves like NumPy arrays, but be careful with indexes
    print(d[1:4])  # Output: [1, 2, 3]

    d[1:3] = [-1, -2]  # Output: [2, 3, 4]
    print(d)  # Output: NumpyDeque([0, -1, -2, 3, 4, 5, 6, 7, 8, 9])

    # Note that values move once maxsize is exceeded
    print(d[2])  # Output: -2
    d.put(10)
    print(d)  # Output: NumpyDeque([-1, -2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(d[2])  # Output: 3
    d.put(11)
    print(d)  # Output: NumpyDeque([-2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    print(d[2])  # Output: 4
    d.putleft(99)
    print(d)  # Output: NumpyDeque([99, -2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(d[2])  # Output: 3

    # Becareful about the size
    d = NumpyDeque(maxsize=5)
    d.put(5)
    d.put(4)
    d.put(3)
    print(d)  # Output: NumpyDeque([5, 4, 3])
    try:
        print(d[3])  # Raises index error!!!
    except IndexError:
        pass
