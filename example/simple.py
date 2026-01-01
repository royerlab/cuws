import numpy as np
import cupy as cp
import napari
import cu1nn
import edt

from cupyx.profiler import benchmark


def main() -> None:
    # Generate an initial image with two overlapping circles
    size = 2 ** 10
    print(f"Size: {size, size}")
    x, y = np.indices((size, size))
    x, y = x / size, y / size
    x1, y1, x2, y2 = 0.3375, 0.3125, 0.55, 0.65
    r1, r2 = 0.1875, 0.25
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    foreground = np.logical_or(mask_circle1, mask_circle2)

    x3, y3 = 0.8, 0.425
    r3 = 0.2
    mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3**2
    foreground = np.logical_or(foreground, mask_circle3)

    distance = edt.edt(foreground).astype(np.float32)

    contour = -distance
    contour = contour - contour.min()
    contour = contour.astype(np.uint16)

    results = benchmark(cu1nn.watershed_from_minima, args=(cp.asarray(contour), cp.asarray(foreground)), n_repeat=5, n_warmup=2)
    gpu_times = results.gpu_times
    cpu_times = results.cpu_times
    print(f"GPU times: {gpu_times.mean():>6.3f}+/-{gpu_times.std():>6.3f} secs")
    print(f"CPU times: {cpu_times.mean():>6.3f}+/-{cpu_times.std():>6.3f} secs")

    labels = cu1nn.watershed_from_minima(cp.asarray(contour), cp.asarray(foreground))
    labels = labels.get()

    uniq_labels = np.unique(labels)[1:] - 1
    y = uniq_labels // labels.shape[1]
    x = uniq_labels % labels.shape[1]

    points = np.stack([y, x], axis=1)
    values = contour[y, x]
    for lb, pt, val in zip(uniq_labels, points, values):
        print(f"{lb}: at {pt} with value {val}")
    
    text = {
        'string': 'Value is {value:.2f}',
        'size': 10,
        'color': 'black',
        'translation': np.array([0, 0]),
    }
    
    viewer = napari.Viewer()
    viewer.add_image(contour)
    viewer.add_labels(labels)
    viewer.add_points(
        points, size=2,
        face_color="white",
        properties={"value": values},
        text=text
    )
    napari.run()



if __name__ == "__main__":
    main()
