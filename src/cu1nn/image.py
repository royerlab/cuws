
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.csgraph import connected_components


preamble = r"""
#define SWAPMIN(values, mask, idx, min_value, min_index) \
    if (mask[idx] && \
        (values[idx] < min_value || \
         (values[idx] == min_value && idx < min_index))) \
    { \
        min_value = values[idx]; \
        min_index = idx; \
    }

#define IDX(z, y, x, height, width) (z * height * width + y * width + x)
"""


# TODO: convert to rawkernel with 3D grid and block
_3d_image_1nn = cp.ElementwiseKernel(
    r"raw T image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw float32 data, raw int64 indices",
    r"""
    if (mask[i])
    {
        long long z = i / (height * width);
        long long y = (i % (height * width)) / width;
        long long x = i % width;

        long long nz, ny, nx, nidx;
        T min_value = image[i];
        long long min_index = i;

        nz = z + 1;
        ny = y;
        nx = x;

        if (nz < depth) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPMIN(image, mask, nidx, min_value, min_index);
        }

        nz = z - 1;
        if (nz >= 0) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPMIN(image, mask, nidx, min_value, min_index);
        }

        nz = z;
        ny = y + 1;
        if (ny < height) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPMIN(image, mask, nidx, min_value, min_index);
        }

        ny = y - 1;
        if (ny >= 0) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPMIN(image, mask, nidx, min_value, min_index);
        }

        ny = y;
        nx = x + 1;
        if (nx < width) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPMIN(image, mask, nidx, min_value, min_index);
        }

        nx = x - 1;
        if (nx >= 0) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPMIN(image, mask, nidx, min_value, min_index);
        }

        indices[i] = min_index;
        data[i] = min_value + 1e-8f; // avoiding zeros
    }
    """,
    r"_3d_image_1nn",
    preamble=preamble,
)


# _find_flat_zones = cp.ElementwiseKernel(
#     r"raw int32 image, raw bool mask, raw int32 labels, raw float32 min_values",
#     """
#     """,
#     r"_find_flat_zones",
# )

_group_by = cp.ElementwiseKernel(
    "raw int32 labels, raw float32 values",
    "raw float32 min_values",
    """
    atomicMin(&min_values[labels[i]], values[i]);
    """,
    "_group_by",
)

def _measure_time(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__:<10} took: {end - start:>6.3f} secs")
        return result
    return wrapper

def watershed_from_minima(
    image: cp.ndarray,
    mask: cp.ndarray,
) -> cp.ndarray:

    orig_shape = image.shape

    if image.ndim == 2:
        image = image[None, ...]
        mask = mask[None, ...]

    if image.shape != mask.shape:
        raise ValueError(f"Image and mask must have the same shape: {image.shape} != {mask.shape}")
    
    size = np.prod(image.shape)

    indptr = cp.arange(size + 1, dtype=cp.int64)
    indices = cp.arange(size, dtype=cp.int64)
    data = cp.zeros(size, dtype=cp.float32)

    flat_mask = mask.ravel()
    flat_image = image.ravel()

    _measure_time(_3d_image_1nn)(flat_image, flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), data, indices, size=size)

    graph = _measure_time(csp.csr_matrix)((data, indices, indptr), shape=(size, size))
    n_cc, cc = _measure_time(connected_components)(graph, directed=True, connection='weak', return_labels=True)

    cc_min_values = cp.full(n_cc, np.inf, dtype=cp.float32)
    _measure_time(_group_by)(cc, data, cc_min_values, size=size)

    # TODO find new graph
    # _find_flat_zones(flat_image, flat_mask, cc, cc_min_values, size=size)
    # n_cc, cc = connected_components(graph, directed=True, connection='weak', return_labels=True)

    cc = cp.where(flat_mask, cc + 1, 0)
    # TODO: use connected_components lower level API, to take of masking and relabel and +1 ourselves

    return cc.reshape(orig_shape)
