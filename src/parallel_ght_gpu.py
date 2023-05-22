# %%writefile sequence_ght_numba.py
import math
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda


THRESHOLD = 200
N_ROTATION_SLICES = 72
MAX_SCALE = 1.4
MIN_SCALE = 0.6
DELTA_SCALE_RATIO = 0.1
N_SCALE_SLICE = int((MAX_SCALE - MIN_SCALE) // DELTA_SCALE_RATIO + 1)
BLOCK_SIZE = 10
THRESHOLD_RATIO = 0.3
DELTA_ROTATION_ANGLE = 360 / N_ROTATION_SLICES
IMAGE_DIR = 'drive/MyDrive/ltss_seminar/images'

# numpy array sobel filter
sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


@cuda.jit
def convert_to_gray(image: np.array, result: np.array):
    r, c = cuda.grid(2)
    if r < image.shape[0] and c < image.shape[1]:
        result[r][c] = 0.299 * image[r][c][0] + 0.587 * image[r][c][1] + 0.114 * image[r][c][2]


@cuda.jit
def convolve(sobel_filter: np.array, gray_src: np.array, result: np.array):
    r, c = cuda.grid(2)
    if r < gray_src.shape[0] and c < gray_src.shape[1]:
        temp = 0
        for jj in range(-1, 2):
            for ii in range(-1, 2):
                jj_pad = max(0, min(gray_src.shape[1] - 1, c + jj))
                ii_pad = max(0, min(gray_src.shape[0] - 1, r + ii))
                temp += gray_src[ii_pad][jj_pad] * sobel_filter[ii + 1][jj + 1]
        result[r][c] = temp


@cuda.jit
def magnitude(magnitude_x: np.array, magnitude_y: np.array, result: np.array):
    r, c = cuda.grid(2)
    if r < magnitude_x.shape[0] and c < magnitude_x.shape[1]:
        result[r][c] = math.sqrt(magnitude_x[r][c] ** 2 + magnitude_y[r][c] ** 2)


@cuda.jit
def orientation(magnitude_x: np.array, magnitude_y: np.array, result: np.array):
    r, c = cuda.grid(2)
    if r < magnitude_x.shape[0] and c < magnitude_x.shape[1]:
        result[r][c] = (math.atan2(magnitude_y[r][c], magnitude_x[r][c]) * 180 / math.pi + 360) % 360


@cuda.jit
def edgemns(magnitude: np.array, orientation: np.array, result: np.array):
    r, c = cuda.grid(2)
    if r < magnitude.shape[0] and c < magnitude.shape[1]:
        pixel_gradient = int(orientation[r][c] // 45) * 45 % 180
        neighbour_one_i = r
        neighbour_one_j = c
        neighbour_two_i = r
        neighbour_two_j = c

        if pixel_gradient == 0:
            neighbour_one_i = r - 1
            neighbour_two_i = r + 1
        elif pixel_gradient == 45:
            neighbour_one_i = r + 1
            neighbour_one_j = c - 1
            neighbour_two_i = r - 1
            neighbour_two_j = c + 1
        elif pixel_gradient == 90:
            neighbour_one_j = c - 1
            neighbour_two_j = c + 1
        elif pixel_gradient == 135:
            neighbour_one_i = r - 1
            neighbour_one_j = c - 1
            neighbour_two_i = r + 1
            neighbour_two_j = c + 1

        neighbour_one_i = max(0, min(magnitude.shape[0] - 1, neighbour_one_i))
        neighbour_one_j = max(0, min(magnitude.shape[1] - 1, neighbour_one_j))
        neighbour_two_i = max(0, min(magnitude.shape[0] - 1, neighbour_two_i))
        neighbour_two_j = max(0, min(magnitude.shape[1] - 1, neighbour_two_j))

        if magnitude[r][c] >= magnitude[neighbour_one_i][neighbour_one_j] and magnitude[r][c] >= magnitude[neighbour_two_i][neighbour_two_j]:
            result[r][c] = magnitude[r][c]
        else:
            result[r][c] = 0


@cuda.jit
def threshold(magnitude: np.array, threshold: int, result: np.array):
    r, c = cuda.grid(2)
    if r < magnitude.shape[0] and c < magnitude.shape[1]:
        if magnitude[r][c] > threshold:
            result[r][c] = 255
        else:
            result[r][c] = 0

@jit(cache=True)
def create_r_table(orientation: np.array, magnitude_threshold: np.array, width_template: int, height_template: int,
                   r_table: list):
    for i in range(orientation.shape[0]):
        for j in range(orientation.shape[1]):
            if magnitude_threshold[i][j] == 255:
                phi = orientation[i][j] % 360
                i_slice = int(phi // DELTA_ROTATION_ANGLE)

                center_x = width_template // 2
                center_y = height_template // 2
                entry_x = center_x - j
                entry_y = center_y - i

                r = math.sqrt(entry_x ** 2 + entry_y ** 2)
                alpha = math.atan2(entry_y, entry_x)

                entry = {'r': r, 'alpha': alpha}
                r_table[i_slice].append(entry)


@jit(cache=True)
def accumulate4D(mag_threshold: np.array, orient: np.array, width_src: int, height_src: int,
                 accumulator: np.array, block_maxima: np.array, r_table: list):
    _max = 0
    for j in range(height_src):
        for i in range(width_src):
            if mag_threshold[j][i] == 255:
                phi = orient[j][i]
                for i_theta in range(N_ROTATION_SLICES):
                    theta = i_theta * DELTA_ROTATION_ANGLE
                    theta_r = math.radians(theta)
                    i_slice = int(((phi - theta + 360) % 360) // DELTA_ROTATION_ANGLE)
                    entries = r_table[i_slice]
                    for entry in entries:
                        r = entry['r']
                        alpha = entry['alpha']
                        for scale in range(N_SCALE_SLICE):
                            s = scale * DELTA_SCALE_RATIO + MIN_SCALE
                            xc = int(i + r * s * math.cos(alpha + theta_r))
                            yc = int(j + r * s * math.sin(alpha + theta_r))

                            if xc < 0 or xc >= width_src or yc < 0 or yc >= height_src:
                                continue
                            accumulator[scale][i_theta][yc // BLOCK_SIZE][xc // BLOCK_SIZE] += 1
                            if accumulator[scale][i_theta][yc // BLOCK_SIZE][xc // BLOCK_SIZE] > \
                                    block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['hits']:
                                block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['hits'] = \
                                    accumulator[scale][i_theta][yc // BLOCK_SIZE][xc // BLOCK_SIZE]
                                block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['x'] = xc
                                block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['y'] = yc
                                if accumulator[scale][i_theta][yc // BLOCK_SIZE][xc // BLOCK_SIZE] > _max:
                                    _max = accumulator[scale][i_theta][yc // BLOCK_SIZE][xc // BLOCK_SIZE]
    maxima_threshold = round(_max * THRESHOLD_RATIO)

    return block_maxima, maxima_threshold


@jit(cache=True)
def accumulate(mag_threshold: np.array, orient: np.array, width_src: int, height_src: int,
                 accumulator: np.array, block_maxima: np.array, r_table: list):
    _max = 0
    for j in range(height_src):
        for i in range(width_src):
            if mag_threshold[j][i] == 255:
                phi = orient[j][i]
                i_slice = int(phi // DELTA_ROTATION_ANGLE)
                entries = r_table[i_slice]
                for entry in entries:
                    r = entry['r']
                    alpha = entry['alpha']
                    xc = int(i + r * math.cos(alpha))
                    yc = int(j + r * math.sin(alpha))

                    if xc < 0 or xc >= width_src or yc < 0 or yc >= height_src:
                        continue
                    accumulator[yc // BLOCK_SIZE][xc // BLOCK_SIZE] += 1
                    block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['hits'] = accumulator[yc // BLOCK_SIZE][
                        xc // BLOCK_SIZE]
                    block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['x'] = xc
                    block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['y'] = yc
                    if accumulator[yc // BLOCK_SIZE][xc // BLOCK_SIZE] > _max:
                        _max = accumulator[yc // BLOCK_SIZE][xc // BLOCK_SIZE]
    maxima_threshold = round(_max * THRESHOLD_RATIO)
    return block_maxima, maxima_threshold


src = cv2.imread(f'{IMAGE_DIR}/leaves.png')
height_src = src.shape[0]
width_src = src.shape[1]
template = cv2.imread(f'{IMAGE_DIR}/leaf.png')
height_template = template.shape[0]
width_template = template.shape[1]
r_table = [[] for _ in range(N_ROTATION_SLICES)]
block_size = (32, 32)
grid_size_template = (math.ceil(height_template / block_size[0]), math.ceil(width_template / block_size[1]))
grid_size_src = (math.ceil(height_src / block_size[0]), math.ceil(width_src / block_size[1]))
wblock = (width_src + BLOCK_SIZE - 1) // BLOCK_SIZE
hblock = (height_src + BLOCK_SIZE - 1) // BLOCK_SIZE

print("----------Start processing template----------\n")
time_process = 0

# Gray convert
gray_template = np.zeros(template.shape[:2], dtype=np.float64)
start = time.time()
convert_to_gray[grid_size_template, block_size](template, gray_template)
end = time.time()
time_process += end - start

# Sobel filter
magnitude_x = np.zeros_like(gray_template)
magnitude_y = np.zeros_like(gray_template)
start = time.time()
convolve[grid_size_template, block_size](sobel_filter_x, gray_template, magnitude_x)
convolve[grid_size_template, block_size](sobel_filter_y, gray_template, magnitude_y)
end = time.time()
time_process += end - start

# Magnitude and orientation
magnitude_tpl = np.zeros_like(gray_template)
orientation_tpl = np.zeros_like(gray_template)
start = time.time()
magnitude[grid_size_template, block_size](magnitude_x, magnitude_y, magnitude_tpl)
orientation[grid_size_template, block_size](magnitude_x, magnitude_y, orientation_tpl)
end = time.time()
time_process += end - start

# Edge minmax
edge_minmax_tpl = np.zeros_like(gray_template)
start = time.time()
edgemns[grid_size_template, block_size](magnitude_tpl, orientation_tpl, edge_minmax_tpl)
end = time.time()
time_process += end - start

# Threshold
mag_threshold_tpl = np.zeros_like(gray_template)
start = time.time()
threshold[grid_size_template, block_size](edge_minmax_tpl, THRESHOLD, mag_threshold_tpl)
end = time.time()
time_process += end - start

# Create R-table
start = time.time()
create_r_table(orientation_tpl, mag_threshold_tpl, width_template, height_template, r_table)
end = time.time()
time_process += end - start

print("----------End processing template----------\n")
print(f"Time processing template: {time_process}\n")


print("----------Start accumulating src----------\n")
time_process = 0


# Gray convert
gray_src = np.zeros(src.shape[:2], dtype=np.float64)
start = time.time()
convert_to_gray[grid_size_src, block_size](src, gray_src)
end = time.time()
time_process += end - start

# Sobel filter
magnitude_x = np.zeros_like(gray_src)
magnitude_y = np.zeros_like(gray_src)
start = time.time()
convolve[grid_size_src, block_size](sobel_filter_x, gray_src, magnitude_x)
convolve[grid_size_src, block_size](sobel_filter_y, gray_src, magnitude_y)
end = time.time()
time_process += end - start

# Magnitude and orientation
magnitude_src = np.zeros_like(gray_src)
orientation_src = np.zeros_like(gray_src)
start = time.time()
magnitude[grid_size_src, block_size](magnitude_x, magnitude_y, magnitude_src)
orientation[grid_size_src, block_size](magnitude_x, magnitude_y, orientation_src)
end = time.time()
time_process += end - start

# Edge minmax
edge_minmax_src = np.zeros_like(gray_src)
start = time.time()
edgemns[grid_size_src, block_size](magnitude_src, orientation_src, edge_minmax_src)
end = time.time()
time_process += end - start

# Threshold
mag_threshold_src = np.zeros_like(gray_src)
start = time.time()
threshold[grid_size_src, block_size](edge_minmax_src, THRESHOLD, mag_threshold_src)
end = time.time()
time_process += end - start

# Accumulate
accumulator = np.zeros((hblock, wblock), dtype=np.int32)
block_maxima = np.zeros((hblock, wblock), dtype=[('x', int), ('y', int), ('hits', int)])
start = time.time()
block_maxima, maxima_threshold = accumulate(mag_threshold_src, orientation_src, width_src, height_src, accumulator, block_maxima, r_table)
end = time.time()
time_process += end - start

# Draw
wblock = (width_src + BLOCK_SIZE - 1) // BLOCK_SIZE
hblock = (height_src + BLOCK_SIZE - 1) // BLOCK_SIZE
plt.imshow(src)
for j in range(hblock):
    for i in range(wblock):
        if block_maxima[j][i]['hits'] > maxima_threshold:
            plt.plot([block_maxima[j][i]['x']], [block_maxima[j][i]['y']], marker='o', color="yellow")

plt.savefig(f'{IMAGE_DIR}/output.png')
plt.show()

print("----------End accumulating src----------\n")
print(f"Time process: {time_process}s\n")