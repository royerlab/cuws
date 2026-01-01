import logging
import time

import cupy as cp
import napari
import zarr

import cu1nn

logging.basicConfig(level=logging.INFO)


def main() -> None:
    path = "/home/jordao/Softwares/ultrack-td/examples/"
    contour = zarr.open(path + "boundaries.zarr")[0]
    foreground = zarr.open(path + "detection.zarr")[0]

    print(f"shape: {contour.shape}")
    print(f"num. elements: {contour.size}")

    start = time.time()
    cu_contour = cp.asarray(contour)
    cu_foreground = cp.asarray(foreground)
    end = time.time()
    print(f"CPU->GPU: {end - start} seconds")

    cu_contour = (cu_contour * (2**16)).astype(cp.uint16)

    # warmup
    print("Executing warmup run...")
    cu1nn.watershed_from_minima(cu_contour, cu_foreground)
    print("done")

    start = time.time()
    labels = cu1nn.watershed_from_minima(cu_contour, cu_foreground)
    end = time.time()
    print(f"watershed: {end - start} seconds")

    start = time.time()
    labels = labels.get()
    end = time.time()
    print(f"GPU->CPU: {end - start} seconds")

    viewer = napari.Viewer()
    viewer.add_image(contour)
    viewer.add_labels(labels)
    napari.run()


if __name__ == "__main__":
    main()
