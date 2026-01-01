
import logging
import numpy as np
import cupy as cp

LOG = logging.getLogger(__name__)


preamble = r"""
#define IDX(z, y, x, height, width) (z * height * width + y * width + x)

typedef unsigned long long ull;

__constant__ int d_size = 26;
__constant__ int dz[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1};
__constant__ int dy[] = {-1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1};
__constant__ int dx[] = {-1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1};
"""


# TODO: convert to rawkernel with 3D grid and block
_3d_image_1nn = cp.ElementwiseKernel(
    r"raw T image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw uint64 indices",
    r"""
    if (mask[i])
    {
        ull z = i / (height * width);
        ull y = (i % (height * width)) / width;
        ull x = i % width;

        ull nz, ny, nx, nidx;
        T min_value = image[i];
        ull min_index = i;

        #pragma unroll
        for (int j = 0; j < d_size; ++j)
        {
            ull nz = z + dz[j];
            ull ny = y + dy[j];
            ull nx = x + dx[j];

            // zero check is not needed since we are using unsigned integers
            if (nz < depth && ny < height && nx < width)
            {
                ull nidx = IDX(nz, ny, nx, height, width);
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
    r"raw uint64 roots",
    r"""
    if (mask[i]) {
        unsigned long long r = roots[i];
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
    r"raw uint64 roots, raw uint64 n_changed",
    r"""
    if (mask[i])
    {
        ull r = roots[i];
        T i_value = image[i];
        T r_value = image[r];

        // this is only possible if the pixel is tied with the root
        if (r_value != i_value) return;

        ull z = i / (height * width);
        ull y = (i % (height * width)) / width;
        ull x = i % width;

        #pragma unroll
        for (int j = 0; j < d_size; ++j)
        {
            ull nz = z + dz[j];
            ull ny = y + dy[j];
            ull nx = x + dx[j];

            if (nz < depth && ny < height && nx < width)
            {
                ull nidx = IDX(nz, ny, nx, height, width);
                if (mask[nidx]) {
                    ull nr = roots[nidx];
                    T nr_value = image[nr];
                    if (image[nidx] == i_value &&  // tie-zone
                        (r_value > nr_value || (r_value == nr_value && r > nr))) // deeper root
                    {
                        r_value = nr_value;
                        r = nr;
                    }
                }
            }
        }
        if (r != roots[i]) {
            atomicAdd(&n_changed[0], 1);
        }
        roots[i] = r;
    }
    """,
    r"_merge_flat_zones",
    preamble=preamble,
)


_relabel_inplace = cp.ElementwiseKernel(
    r"bool mask", r"uint64 roots",
    r"""
    roots = (mask) ? roots + 1 : 0;
    """,
    r"_relabel_inplace",
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

    roots = cp.arange(size, dtype=cp.uint64)

    flat_mask = mask.ravel()
    flat_image = image.ravel()

    _3d_image_1nn(flat_image, flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), roots, size=size)

    n_iters = 0
    n_changed = cp.ones(1, dtype=cp.uint64)

    while n_changed[0] > 0:
        _assign_root(flat_mask, roots, size=size)
        n_changed[0] = 0
        _merge_flat_zones(flat_image, flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), roots, n_changed, size=size)
        n_iters += 1

    LOG.info("Performed %d merge-flat-zones iterations", n_iters)

    _relabel_inplace(flat_mask, roots)

    return roots.reshape(orig_shape)
