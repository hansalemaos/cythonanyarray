# Pass arrays of any dimension to Cython as memory views (no Global Interpreter Lock (GIL))

## pip install cythonanyarray

### Tested against Windows / Python 3.11 / Anaconda


## Cython (and a C/C++ compiler) must be installed

This module provides utility functions for creating and manipulating flattened indices and pointers for multi-dimensional arrays in Cython. It is designed to facilitate passing arrays of any dimension to Cython as memory views, enabling the release of the Global Interpreter Lock (GIL) for improved performance in parallel processing.


```python
# Functions:
- get_pointer_array: Get a flat pointer array from the original array. # if you change it, the original data changes too
- get_iterarray: Get an array of flattened indices along with a flat pointer array for an input array.
- get_flat_iter_for_cython: Get an array of flattened indices and a flat pointer array suitable for use in Cython.
- get_iterarray_shape: Get an array of flattened indices with a specified last dimension.

import cv2
import numpy as np
from cythonanyarray import get_flat_iter_for_cython

data = cv2.imread(r"C:\Users\hansc\Pictures\socialmediaiconsmatching.png")
indexarray, flatpointerarray = get_flat_iter_for_cython(data, dtype=np.int64, unordered=True)
print(flatpointerarray)
# [[      0       0       0       0]
#  [   5745       1       0       0]
#  [  11490       2       0       0]
#  ...
#  [5543924     964    1914       2]
#  [5549669     965    1914       2]
#  [5555414     966    1914       2]]
# [51 51 51 ... 43 43 43]
# Iterate through the flattened indices - Blueprint for Cython
# This snipped is crap in Python but in Cython it is pure gold (don't forget to add types!)
flatiter = len(indexarray)
for i in range(flatiter):
    if flatpointerarray[indexarray[i][0]] == 255:
        print(indexarray[i][1:])
        break

# Accessing the original array using the flattened indices
print(data[34, 0, 0])  # Output: [34, 0, 0]

# Another example with a multi-dimensional array
data = np.arange(10 * 4 * 12 * 12 * 32).reshape((10, 4, 12, 12, 32)).astype(np.float64)
indexarray, flatpointerarray = get_flat_iter_for_cython(data, dtype=np.int64, unordered=True)
print(flatpointerarray)
# [[     0      0      0      0      0      0]
#  [ 18432      1      0      0      0      0]
#  [ 36864      2      0      0      0      0]
#  ...
#  [147455      7      3     11     11     31]
#  [165887      8      3     11     11     31]
#  [184319      9      3     11     11     31]]
# [0.00000e+00 1.00000e+00 2.00000e+00 ... 1.84317e+05 1.84318e+05
#  1.84319e+05]
# Iterate through the flattened indices
flatiter = len(indexarray)
for i in range(flatiter):
    if flatpointerarray[indexarray[i][0]] == 255.0:
        print(indexarray[i][1:])
        break

# Accessing the original array using the flattened indices
# Output: [0, 0, 0, 7, 31]


# Changing values

data = np.arange(100 * 100 * 100).reshape((100, 100, 100)).astype(np.int32)
indexarray, flatpointerarray = get_flat_iter_for_cython(data, dtype=np.int64, unordered=True)
print(flatpointerarray)
flatiter = len(indexarray)
for i in range(flatiter):
    if indexarray[i][3] % 2 == 0:
        flatpointerarray[indexarray[i][0]] = 10000000
print(data)
# [[[10000000        1 10000000 ...       97 10000000       99]
#   [10000000      101 10000000 ...      197 10000000      199]
#   [10000000      201 10000000 ...      297 10000000      299]
#   ...
#   [10000000     9701 10000000 ...     9797 10000000     9799]
#   [10000000     9801 10000000 ...     9897 10000000     9899]
#   [10000000     9901 10000000 ...     9997 10000000     9999]]

```