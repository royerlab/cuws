import logging

import cupy as cp
import numpy as np

LOG = logging.getLogger(__name__)


preamble = r"""
#define IDX(z, y, x, height, width) (z * height * width + y * width + x)
#define IDX_MASK (~(0xFFFFULL << 48))
#define VAL_IDX(val, idx) (idx | ((ull) val << 48))

typedef unsigned long long ull;

__constant__ int d_size = 26;
__constant__ int dz[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1
};
__constant__ int dy[] = {
    -1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1
};
__constant__ int dx[] = {
    -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1
};
"""

_3d_image_1nn_str = r"""
    ull z = i / (height * width);
    ull y = (i % (height * width)) / width;
    ull x = i % width;

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

    roots[i] = min_value;
"""

_3d_image_1nn = cp.ElementwiseKernel(
    r"raw uint16 image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw uint64 roots",
    f"if (mask[i]) {{\n{_3d_image_1nn_str}\n}}\n",
    r"_3d_image_1nn",
    preamble=preamble,
)

_3d_image_1nn_sparse = cp.ElementwiseKernel(
    r"raw int64 indices, raw uint16 image, raw bool mask, int64 depth, int64 height, int64 width",
    r"raw uint64 roots",
    f"i = indices[i];\n{_3d_image_1nn_str}\n",
    r"_3d_image_1nn_sparse",
    preamble=preamble,
)

_assign_root_str = r"""
    ull next, r = roots[i];
    while (r != (next = roots[r & IDX_MASK])) {
        r = next;
    }
    roots[i] = r;
"""

_assign_root = cp.ElementwiseKernel(
    r"raw bool mask",
    r"raw uint64 roots",
    f"if (mask[i]) {{\n{_assign_root_str}\n}}\n",
    r"_assign_root",
    preamble=preamble,
)

_assign_root_sparse = cp.ElementwiseKernel(
    r"raw int64 indices, raw bool mask",
    r"raw uint64 roots",
    f"i = indices[i];\n{_assign_root_str}\n",
    r"_assign_root_sparse",
    preamble=preamble,
)

_merge_flat_zones_str = r"""
    ull r = roots[i];
    ull root_idx = r & IDX_MASK;
    ull i_value = image[i];
    ull r_value = r >> 48;

    // we skip pixels that are outside the dynamic height range
    if (i_value - r_value > h) return;

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
                if (nr < r) { // deeper root
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
        atomicAdd(&changed[0], 1);
    }
"""

_merge_flat_zones = cp.ElementwiseKernel(
    r"raw uint16 image, raw bool mask, int64 h, int64 depth, int64 height, int64 width",
    r"raw uint64 roots, raw uint64 changed",
    f"if (mask[i]) {{\n{_merge_flat_zones_str}\n}}\n",
    r"_merge_flat_zones",
    preamble=preamble,
)

_merge_flat_zones_sparse = cp.ElementwiseKernel(
    r"raw int64 indices, raw uint16 image, raw bool mask, int64 h, int64 depth, int64 height, int64 width",
    r"raw uint64 roots, raw uint64 changed",
    f"i = indices[i];\n{_merge_flat_zones_str}\n",
    r"_merge_flat_zones_sparse",
    preamble=preamble,
)

_relabel_inplace = cp.ElementwiseKernel(
    r"bool mask",
    r"uint64 roots",
    r"""
    roots = (mask) ? (roots & IDX_MASK) + 1 : 0;
    """,
    r"_relabel_inplace",
    preamble=preamble,
)


def watershed_from_minima(
    image: cp.ndarray,
    mask: cp.ndarray,
    h: int = 0,
    sparse: bool = True,
) -> cp.ndarray:
    """
    Watershed segmentation from minima.

    Parameters
    ----------
    image : cp.ndarray
        Grayscale image indicating the object boundaries, must be of type 'uint16'.
    mask : cp.ndarray
        Binary mask indicating the pixels that are part of the foreground.
    h : int, optional
        Watershed dynamic merging height, by default 0.
        If the pixel intensity difference between the root and the boundary pixels are less than `h`,
        they are merged to the deepest neighboring root
        FIXME: there might be a bug here, should it merge to the deepest root or to the region with weakest boundary?
    sparse : bool, optional
        Whether to use sparse kernel, by default True.
        Sparse kernels are faster depending on the number of foreground pixels.
        It uses a bit more memory than the dense kernels because it needs to store the indices of the foreground pixels.

    Returns
    -------
    cp.ndarray
        Integer labels image with the same shape as the input image.
        Background is labeled as 0.
    """
    if image.ndim != 2 and image.ndim != 3:
        raise ValueError(f"Image must be 2D or 3D, got '{image.ndim}' dimensions")

    orig_shape = image.shape

    if image.ndim == 2:
        image = image[None, ...]
        mask = mask[None, ...]

    if image.shape != mask.shape:
        raise ValueError(f"Image and mask must have the same shape: {image.shape} != {mask.shape}")

    if image.dtype != np.uint16:
        raise ValueError(f"Image must be of type 'uint16', got '{image.dtype}'")

    size = np.prod(image.shape)
    if size > 2**48:
        raise ValueError(f"Size '{size}' is larger than the maximum supported size of '2 ** 48' ({2**48})")

    roots = cp.arange(size, dtype=cp.uint64)

    flat_mask = mask.ravel()
    flat_image = image.ravel()

    # TODO:
    # - reduce branching of kernels by having different branch for border and non-border pixels
    # - reduce number of divisions by using a 3D grid kernel
    changed = cp.ones(1, dtype=cp.uint64)
    n_iters = 0

    if sparse:
        non_zero_indices = cp.nonzero(flat_mask)[0]
        non_zero_size = non_zero_indices.size

        _3d_image_1nn_sparse(
            non_zero_indices,
            flat_image,
            flat_mask,
            int(image.shape[0]),
            int(image.shape[1]),
            int(image.shape[2]),
            roots,
            size=non_zero_size,
        )

        while changed[0]:
            _assign_root_sparse(non_zero_indices, flat_mask, roots, size=non_zero_size)
            changed[0] = 0
            _merge_flat_zones_sparse(
                non_zero_indices,
                flat_image,
                flat_mask,
                h,
                int(image.shape[0]),
                int(image.shape[1]),
                int(image.shape[2]),
                roots,
                changed,
                size=non_zero_size,
            )
            n_iters += 1

    else:
        _3d_image_1nn(
            flat_image,
            flat_mask,
            int(image.shape[0]),
            int(image.shape[1]),
            int(image.shape[2]),
            roots,
            size=size,
        )
        while changed[0]:
            _assign_root(flat_mask, roots, size=size)
            changed[0] = 0
            _merge_flat_zones(
                flat_image,
                flat_mask,
                h,
                int(image.shape[0]),
                int(image.shape[1]),
                int(image.shape[2]),
                roots,
                changed,
                size=size,
            )
            n_iters += 1

    LOG.info("Performed %d minima merging iterations", n_iters)

    _relabel_inplace(flat_mask, roots)

    return roots.reshape(orig_shape)
