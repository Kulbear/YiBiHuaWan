import numpy as np
import cv2

START_LABEL = 0


def crop_image(img):
    mask = img == 0

    col_tol = int(mask.shape[0] * 0.1)
    row_tol = int(mask.shape[1] * 0.1)

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # now let's do some filter work
    for col in range(x1 - 1, x0, -1):
        non_zero = np.sum(mask[:, col] > 0)
        if non_zero < col_tol:
            x1 = col
        else:
            break

    for row in range(y1 - 1, y0, -1):
        non_zero = np.sum(mask[row, :] > 0)
        if non_zero < row_tol:
            y1 = row
        else:
            break

    # Get the contents of the bounding box.
    cropped = img[y0:y1, x0:x1]
    return cropped


def generate_grid(row, col):
    return np.array([[None for _ in range(col)] for _ in range(row)])


def fill_grid(img, grid, start=(0, 0)):
    counter = 1
    row, col = grid.shape
    grid[start[0]][start[1]] = START_LABEL
    for i in range(row):
        for j in range(col):
            if img[(i + 1) * 20 - 10][(j + 1) * 20 - 10] == 0 and grid[i][j] != START_LABEL:
                grid[i][j] = counter
                counter += 1
    return grid


def img2grid(row, col, img_path='test.jpeg', start=(0, 0)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = (img >= 240).astype(np.uint8)

    img = crop_image(img)

    cell_size = 20
    img = cv2.resize(img, (col * cell_size, row * cell_size))

    grid = generate_grid(row, col)
    grid = fill_grid(img, grid, start)
    # grid = mark_vertices(grid)
    return grid, img
