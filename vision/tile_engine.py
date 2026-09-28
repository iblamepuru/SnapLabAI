import cv2
import numpy as np

def create_tiles(image, tile_size=640, overlap=0.25):
    h, w = image.shape[:2]

    if h <= tile_size and w <= tile_size:
        return [(image, 0, 0)]

    step = int(tile_size * (1.0 - overlap))

    tiles = []

    ys = list(range(0, max(h - tile_size, 0) + 1, step))
    xs = list(range(0, max(w - tile_size, 0) + 1, step))

    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))

    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))

    for y in ys:
        for x in xs:
            crop = image[
                y:min(y + tile_size, h),
                x:min(x + tile_size, w)
            ]

            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                padded = np.zeros(
                    (tile_size, tile_size, 3),
                    dtype=image.dtype
                )

                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded

            tiles.append((crop, x, y))

    return tiles


def translate_detection(box, offset_x, offset_y):
    x, y, w, h = box

    return [
        float(x + offset_x),
        float(y + offset_y),
        float(w),
        float(h)
    ]
