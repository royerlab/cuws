
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.csgraph import connected_components


preamble = r"""
#define IDX(z, y, x, height, width) (z * height * width + y * width + x)

typedef long long ll;

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
        ll z = i / (height * width);
        ll y = (i % (height * width)) / width;
        ll x = i % width;

        ll nz, ny, nx, nidx;
        T min_value = image[i];
        ll min_index = i;

        const ll dz[] = {1, -1, 0, 0, 0, 0};
        const ll dy[] = {0, 0, 1, -1, 0, 0};
        const ll dx[] = {0, 0, 0, 0, 1, -1};

        #pragma unroll
        for (int i = 0; i < 6; ++i)
        {
            ll nz = z + dz[i];
            ll ny = y + dy[i];
            ll nx = x + dx[i];

            if (nz >= 0 && nz < depth && ny >= 0 && ny < height && nx >= 0 && nx < width)
            {
                ll nidx = IDX(nz, ny, nx, height, width);
                if (mask[nidx] && (image[nidx] < min_value ||
                                   (image[nidx] == min_value && nidx < min_index)))
                {
                    min_value = image[nidx];
                    min_index = nidx;
                }
            }
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
        ll z = i / (height * width);
        ll y = (i % (height * width)) / width;
        ll x = i % width;
        ll r = roots[i];

        T i_value = image[i];
        T r_value = image[r];

        ll nz, ny, nx, nidx;
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
