import numpy as np
import cupy as cp
import napari
import cu1nn
import edt


def main() -> None:
    # Generate an initial image with two overlapping circles
    x, y = np.indices((80, 80))
    x1, y1, x2, y2 = 27, 25, 44, 52
    r1, r2 = 15, 20
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    # mask_circle1 = np.zeros_like(mask_circle1)  # FIXME
    foreground = np.logical_or(mask_circle1, mask_circle2)

    x3, y3 = 64, 34
    r3 = 16
    mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3**2
    foreground = np.logical_or(foreground, mask_circle3)

    distance = edt.edt(foreground).astype(np.float32)

    foreground = foreground[None, ...]
    distance = distance[None, ...]

    contour = -distance
    contour = contour - contour.min()
    contour = contour.astype(np.float32)

    labels = cu1nn.watershed_from_minima(cp.asarray(contour), cp.asarray(foreground))
    labels = labels.get()
    
    viewer = napari.Viewer()
    viewer.add_image(contour)
    viewer.add_labels(labels)
    napari.run()



if __name__ == "__main__":
    main()
