
import logging
import numpy as np
import cupy as cp

LOG = logging.getLogger(__name__)


preamble = r"""
#define IDX(z, y, x, height, width) (z * height * width + y * width + x)
#define IDX_MASK (~(0xFFFFULL << 48))
#define VAL_IDX(val, idx) (idx | ((ull) val << 48))

typedef unsigned long long ull;

__constant__ int d_size = 26;
__constant__ int dz[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1};
__constant__ int dy[] = {-1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1};
__constant__ int dx[] = {-1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1};
"""


_3d_image_1nn = cp.ElementwiseKernel(
    r"raw uint16 image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw uint64 indices",
    r"""
    if (mask[i])
    {
        ull z = i / (height * width);
        ull y = (i % (height * width)) / width;
        ull x = i % width;

        ull nz, ny, nx, nidx;
        ull min_value = VAL_IDX(image[i], i);

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
                if (mask[nidx]) {
                    ull nval = VAL_IDX(image[nidx], nidx);
                    if (nval < min_value) {
                        min_value = nval;
                    }
                }
            }
        }

        indices[i] = min_value;
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
        ull next, r = roots[i];
        while (r != (next = roots[r & IDX_MASK])) {
            r = next;
        }
        roots[i] = r;
    }
    """,
    r"_assign_root",
    preamble=preamble,
)


_merge_flat_zones = cp.ElementwiseKernel(
    r"raw uint16 image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw uint64 roots",
    r"""
    if (mask[i])
    {
        ull r = roots[i];
        ull root_idx = r & IDX_MASK;
        ull i_value = image[i];
        ull r_value = r >> 48;

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
                    if (image[nidx] == i_value &&  // tie-zone
                        r > nr) // deeper root
                    {
                        r = nr;
                    }
                }
            }
        }
        if (r != roots[i]) {
            if (root_idx != i) {
                roots[i] = r; // using atomicMin only when necessary
            }
            atomicMin(&roots[root_idx], r);
        }
    }
    """,
    r"_merge_flat_zones",
    preamble=preamble,
)


_relabel_inplace = cp.ElementwiseKernel(
    r"bool mask", r"uint64 roots",
    r"""
    roots = (mask) ? (roots & IDX_MASK) + 1 : 0;
    """,
    r"_relabel_inplace",
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
    
    if image.dtype != np.uint16:
        raise ValueError(f"Image must be of type 'uint16', got '{image.dtype}'")
    
    size = np.prod(image.shape)
    if size > 2 ** 48:
        raise ValueError(f"Size '{size}' is larger than the maximum supported size of '2 ** 48' ({2 ** 48})")

    roots = cp.arange(size, dtype=cp.uint64)

    flat_mask = mask.ravel()
    flat_image = image.ravel()

    # TODO: compute non-zero indices and launch kernels only on those
    _3d_image_1nn(flat_image, flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), roots, size=size)
    _assign_root(flat_mask, roots, size=size)

    _merge_flat_zones(flat_image, flat_mask, int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), roots, size=size)
    _assign_root(flat_mask, roots, size=size)

    _relabel_inplace(flat_mask, roots)

    return roots.reshape(orig_shape)
