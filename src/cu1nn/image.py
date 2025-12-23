
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.csgraph import connected_components


_3d_image_1nn = cp.ElementwiseKernel(
    "raw T image, raw bool mask, int64 depth, int64 height, int64 width",
    "raw float32 data, raw int64 indices",
    """
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
            nidx = nz * height * width + ny * width + nx;
            if (image[nidx] < min_value) {
                min_value = image[nidx];
                min_index = nidx;
            }
        }

        nz = z - 1;
        if (nz >= 0) {
            nidx = nz * height * width + ny * width + nx;
            if (image[nidx] < min_value) {
                min_value = image[nidx];
                min_index = nidx;
            }
        }

        nz = z;
        ny = y + 1;
        if (ny < height) {
            nidx = nz * height * width + ny * width + nx;
            if (image[nidx] < min_value) {
                min_value = image[nidx];
                min_index = nidx;
            }
        }

        ny = y - 1;
        if (ny >= 0) {
            nidx = nz * height * width + ny * width + nx;
            if (image[nidx] < min_value) {
                min_value = image[nidx];
                min_index = nidx;
            }
        }

        ny = y;
        nx = x + 1;
        if (nx < width) {
            nidx = nz * height * width + ny * width + nx;
            if (image[nidx] < min_value) {
                min_value = image[nidx];
                min_index = nidx;
            }
        }

        nx = x - 1;
        if (nx >= 0) {
            nidx = nz * height * width + ny * width + nx;
            if (image[nidx] < min_value) {
                min_value = image[nidx];
                min_index = nidx;
            }
        }

        indices[i] = min_index;
        data[i] = min_value + 1e-8f; // avoiding zeros
    }
    """,
    "_3d_image_1nn",
)


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

    _3d_image_1nn(image.ravel(), flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), data, indices, size=size)

    sparse_image = csp.csr_matrix((data, indices, indptr), shape=(size, size))

    _, cc = connected_components(sparse_image, directed=True, connection='weak', return_labels=True)

    cc = cp.where(flat_mask, cc + 1, 0)
    # TODO: use connected_components lower level API, to take of masking and relabel and +1 ourselves

    return cc.reshape(orig_shape)
