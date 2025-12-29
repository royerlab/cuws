
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.csgraph import connected_components


preamble = r"""
#define SWAPMIN(values, mask, idx, min_value, min_index) \
{ \
    if (mask[idx] && \
        (values[idx] < min_value || \
         (values[idx] == min_value && idx < min_index))) \
    { \
        min_value = values[idx]; \
        min_index = idx; \
    } \
}

#define IDX(z, y, x, height, width) (z * height * width + y * width + x)

#define SWAPROOT(roots, values, ri_value, r, i, nidx) \
{ \
    long long nr = roots[nidx]; \
    T nv = values[nr]; \
    if (values[nidx] == ri_value && \
        (ri_value > nv || \
         (ri_value == nv && r > nr))) \
    { \
        ri_value = nv; \
        r = nr; \
        roots[i] = nr; \
    } \
}
"""


# TODO: convert to rawkernel with 3D grid and block
_3d_image_1nn = cp.ElementwiseKernel(
    r"raw T image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw int64 indices",
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
    }
    """,
    r"_3d_image_1nn",
    preamble=preamble,
)

_assign_root = cp.ElementwiseKernel(
    r"raw bool mask",
    r"raw int64 roots",
    r"""
    if (mask[i]) {
        long long r = roots[i];
        while (r != roots[r]) {
            r = roots[r];
        }
        roots[i] = r;
    } else {
        roots[i] = -1;
    }
    """,
    r"_assign_root",
)


_merge_flat_zones = cp.ElementwiseKernel(
    r"raw T image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw int64 roots",
    r"""
    if (mask[i])
    {
        long long z = i / (height * width);
        long long y = (i % (height * width)) / width;
        long long x = i % width;
        long long r = roots[i];

        long long nz, ny, nx, nidx;
        T i_value = image[i];
        T r_value = image[r];

        nz = z + 1;
        ny = y;
        nx = x;

        if (nz < depth) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPROOT(roots, image, r_value, r, i, nidx);
        }

        nz = z - 1;
        if (nz >= 0) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPROOT(roots, image, r_value, r, i, nidx);
        }

        nz = z;
        ny = y + 1;
        if (ny < height) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPROOT(roots, image, r_value, r, i, nidx);
        }

        ny = y - 1;
        if (ny >= 0) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPROOT(roots, image, r_value, r, i, nidx);
        }

        ny = y;
        nx = x + 1;
        if (nx < width) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPROOT(roots, image, r_value, r, i, nidx);
        }

        nx = x - 1;
        if (nx >= 0) {
            nidx = IDX(nz, ny, nx, height, width);
            SWAPROOT(roots, image, r_value, r, i, nidx);
        }
    }
    """,
    r"_merge_flat_zones",
    preamble=preamble,
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

    indices = cp.arange(size, dtype=cp.int64)

    flat_mask = mask.ravel()
    flat_image = image.ravel()

    _3d_image_1nn(flat_image, flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), indices, size=size)

    roots = indices.copy()
    _assign_root(flat_mask, roots, size=size)

    _merge_flat_zones(flat_image, flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), roots, size=size)
    _assign_root(flat_mask, roots, size=size)

    return roots.reshape(orig_shape)
