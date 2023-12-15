
import copy
import functools
import warnings



K = 5  # default number of classes in any map scheme with this as an argument
SEEDRANGE = 1000000  # range for drawing random ints from for Natural Breaks


FMT = "{:.2f}"






class MockNumpy(object):
    def __init__(self, int_type=None, float_type=None):
        self.int32 = int_type or int
        self.float32 = float_type or float

        self.inf = self.float32("inf")

    def zeros(self, dims, dtype=int):
        if len(dims) == 1:
            zero = dtype(0)
            return [zero for __ in range(dims[0])]

        return [self.zeros(dims[1:], dtype) for __ in range(dims[0])]

    @staticmethod
    def delete(arr, index):
        return arr[:index] + arr[index + 1 :]







try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(_type):  # noqa ARG001
        def decorator_njit(func):
            @functools.wraps(func)
            def wrapper_decorator(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper_decorator

        return decorator_njit



@njit("f8[:](f8[:], u2)")
def _fisher_jenks_means(values, classes=5):
    """
    Jenks Optimal (Natural Breaks) algorithm implemented in Python.

    Notes
    -----

    The original Python code comes from here:
    http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
    and is based on a JAVA and Fortran code available here:
    https://stat.ethz.ch/pipermail/r-sig-geo/2006-March/000811.html

    Returns class breaks such that classes are internally homogeneous while
    assuring heterogeneity among classes.

    """
    n_data = len(values)
    mat1 = np.zeros((n_data + 1, classes + 1), dtype=np.int32)
    mat2 = np.zeros((n_data + 1, classes + 1), dtype=np.float32)
    mat1[1, 1:] = 1
    mat2[2:, 1:] = np.inf

    v = np.float32(0)
    for _l in range(2, len(values) + 1):
        s1 = np.float32(0)
        s2 = np.float32(0)
        w = np.float32(0)
        for m in range(1, _l + 1):
            i3 = _l - m + 1
            val = np.float32(values[i3 - 1])
            s2 += val * val
            s1 += val
            w += np.float32(1)
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, classes + 1):
                    if mat2[_l, j] >= (v + mat2[i4, j - 1]):
                        mat1[_l, j] = i3
                        mat2[_l, j] = v + mat2[i4, j - 1]
        mat1[_l, 1] = 1
        mat2[_l, 1] = v

    k = len(values)

    kclass = np.zeros(classes + 1, dtype=values.dtype)
    kclass[classes] = values[len(values) - 1]
    kclass[0] = values[0]
    for countNum in range(classes, 1, -1):
        pivot = mat1[k, countNum]
        _id = int(pivot - 2)
        kclass[countNum - 1] = values[_id]
        k = int(pivot - 1)
    return np.delete(kclass, 0)


def _fisher_jenks_means_without_numpy(
    values,
    classes=5,
    np = None
    ):
    """
    As for _fisher_jenks_means above, to keep the code as far as possible
    exactly the same, except with np passable in as a dependency, and with
    matrix[i, j] replaced with matrix[i][j] for speed.


    Jenks Optimal (Natural Breaks) algorithm implemented in Python.

    Notes
    -----

    The original Python code comes from here:
    http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
    and is based on a JAVA and Fortran code available here:
    https://stat.ethz.ch/pipermail/r-sig-geo/2006-March/000811.html



    """
    if np is None:
        np = MockNumpy()

    n_data = len(values)
    mat1 = np.zeros((n_data + 1, classes + 1), dtype=np.int32)
    mat2 = np.zeros((n_data + 1, classes + 1), dtype=np.float32)
    
    for j in range(1, classes + 1):
        mat1[1][j] = 1
        for i in range(2, n_data+1):
            mat2[i][j] = np.inf
    v = 0
    for _l in range(2, len(values) + 1):
        s1 = 0
        s2 = 0
        w = 0
        for m in range(1, _l + 1):
            i3 = _l - m + 1
            val = values[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / np.float32(w)
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, classes + 1):
                    if mat2[_l][j] >= (v + mat2[i4][j - 1]):
                        mat1[_l][j] = i3
                        mat2[_l][j] = v + mat2[i4][j - 1]

        mat1[_l][1] = 1
        mat2[_l][1] = v

    k = len(values)

    kclass = np.zeros((classes + 1,), dtype=type(values[0]))
    kclass[classes] = values[len(values) - 1]
    kclass[0] = values[0]
    for countNum in range(classes, 1, -1):
        pivot = mat1[k][countNum]
        _id = int(pivot - 2)
        kclass[countNum - 1] = values[_id]
        k = int(pivot - 1)
    return np.delete(kclass, 0)


