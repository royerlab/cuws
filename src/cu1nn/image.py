
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp


_3d_image_1nn = cp.ElementwiseKernel(
    "T image, int64 depth, int64, height, int64 width, raw int32 indices",
    "",
    """
    int64 z = i / (height * width);
    int64 y = (i % (height * width)) / width;
    int64 x = i % width;

    int64 nz, ny, nx, nidx;
    T min_value = image[i];
    int64 min_index = i;

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

    indptr = cp.arange(image.shape[0] + 1, dtype=cp.int32)
    indices = cp.zeros(size, dtype=cp.int32)

    _3d_image_1nn(image, image.shape[1], image.shape[2], image.shape[3], indices)

    data = cp.ones(size, dtype=cp.int32)
    sparse_image = csp.csr_matrix((data, indices, indptr), shape=(size, size))

    _, cc = csp.connected_components(sparse_image, directed=True, connection='weak', return_labels=True )

    return cc.reshape(orig_shape)
