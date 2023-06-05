# %%writefile parallel_ght_gpu.py
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
PHI_R_TABLE_INDEX = 0
R_R_TABLE_INDEX = 1
ALPHA_R_TABLE_INDEX = 2
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


@cuda.jit
def create_r_table(orientation: np.array, magnitude_threshold: np.array, width_template: int, height_template: int,
                   r_table: np.array):
    r, c = cuda.grid(2)
    if r < orientation.shape[0] and c < orientation.shape[1]:
        if magnitude_threshold[r][c] == 255:
            phi = orientation[r][c] % 360
            i_slice = int(phi // DELTA_ROTATION_ANGLE)

            center_x = width_template // 2
            center_y = height_template // 2
            entry_x = center_x - c
            entry_y = center_y - r

            _r = math.sqrt(entry_x ** 2 + entry_y ** 2)
            alpha = math.atan2(entry_y, entry_x)

            r_table[r, c, PHI_R_TABLE_INDEX] = i_slice
            r_table[r, c, R_R_TABLE_INDEX] = _r
            r_table[r, c, ALPHA_R_TABLE_INDEX] = alpha
        else:
            r_table[r, c, PHI_R_TABLE_INDEX] = -1

@cuda.jit
def num_edge_pixels_and_convert_r_table_to_1D(magnitude_threshold: np.array, _num_edge_pixels: int, r_table_1D: np.array, r_table: np.array):
    r, c = cuda.grid(2)
    if r < magnitude_threshold.shape[0] and c < magnitude_threshold.shape[1]:
        if magnitude_threshold[r][c] == 255:
            index = cuda.atomic.add(_num_edge_pixels, 0, 1)
            r_table_1D[index, PHI_R_TABLE_INDEX] = r_table[r, c, PHI_R_TABLE_INDEX]
            r_table_1D[index, R_R_TABLE_INDEX] = r_table[r, c, R_R_TABLE_INDEX]
            r_table_1D[index, ALPHA_R_TABLE_INDEX] = r_table[r, c, ALPHA_R_TABLE_INDEX]

@cuda.jit
def num_edge_pixels_and_get_pixel_index(magnitude_threshold: np.array, _num_edge_pixels: int, edge_pixels_1D: np.array):
    r, c = cuda.grid(2)
    if r < magnitude_threshold.shape[0] and c < magnitude_threshold.shape[1]:
        if magnitude_threshold[r][c] == 255:
            index = cuda.atomic.add(_num_edge_pixels, 0, 1)
            edge_pixels_1D[index, 0] = r
            edge_pixels_1D[index, 1] = c


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


@cuda.jit
def accumulate(edge_pixels: np.array, num_edge_pixels: int, orient: np.array, width_src: int, height_src: int,
                 accumulator: np.array, r_table: list):
    index = cuda.grid(1)
    if (index >= num_edge_pixels):
        return

    row = edge_pixels[index, 0]
    col = edge_pixels[index, 1]

    phi = orient[row, col]
    i_slice = int(phi // DELTA_ROTATION_ANGLE)
    for entry in r_table:
        if int(entry[PHI_R_TABLE_INDEX]) == i_slice:
            r = entry[R_R_TABLE_INDEX]
            alpha = entry[ALPHA_R_TABLE_INDEX]
            xc = int(col + r * math.cos(alpha))
            yc = int(row + r * math.sin(alpha))

            if xc < 0 or xc >= width_src or yc < 0 or yc >= height_src:
                continue
            accumulator_index = (yc // BLOCK_SIZE, xc // BLOCK_SIZE)
            value_accum = cuda.atomic.add(accumulator, accumulator_index, 1)


src = cv2.imread(f'{IMAGE_DIR}/leaves.png')
height_src = src.shape[0]
width_src = src.shape[1]
template = cv2.imread(f'{IMAGE_DIR}/leaf.png')
height_template = template.shape[0]
width_template = template.shape[1]
r_table = np.zeros((height_template, width_template, 3), dtype=np.float64)
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
mag_threshold_tpl = np.zeros_like(gray_template, dtype=np.int32)
start = time.time()
threshold[grid_size_template, block_size](edge_minmax_tpl, THRESHOLD, mag_threshold_tpl)
end = time.time()
time_process += end - start

# Create R-tablek
start = time.time()
create_r_table[grid_size_template, block_size](orientation_tpl, mag_threshold_tpl, width_template, height_template,
                                               r_table)
end = time.time()
time_process += end - start

# Convert 2D R_table to 1D R_table
_num_edge_pixels_tpl_np = np.zeros(1, dtype=np.int32)
_num_edge_pixels_tpl = cuda.to_device(_num_edge_pixels_tpl_np)
r_table_1D = np.zeros((height_template * width_template, 3), dtype=np.float64)
num_edge_pixels_and_convert_r_table_to_1D[grid_size_template, block_size](mag_threshold_tpl, _num_edge_pixels_tpl, r_table_1D, r_table)
num_edge_pixels_tpl = _num_edge_pixels_tpl.copy_to_host()[0]
r_table_1D = r_table_1D[:num_edge_pixels_tpl]

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
mag_threshold_src = np.zeros_like(gray_src, dtype=np.int32)
start = time.time()
threshold[grid_size_src, block_size](edge_minmax_src, THRESHOLD, mag_threshold_src)
end = time.time()
time_process += end - start

# Edge pixels
_num_edge_pixels_src_np = np.zeros(1, dtype=np.int32)
_num_edge_pixels_src = cuda.to_device(_num_edge_pixels_src_np)
edge_pixel_src = np.zeros((height_src * width_src, 2), dtype=np.int32)
num_edge_pixels_and_get_pixel_index[grid_size_src, block_size](mag_threshold_src, _num_edge_pixels_src, edge_pixel_src)
num_edge_pixels_src = _num_edge_pixels_src.copy_to_host()[0]
edge_pixel_src = edge_pixel_src[:num_edge_pixels_src]

# Accumulate
_accumulator = np.zeros((hblock, wblock), dtype=np.int32)
accumulator = cuda.to_device(_accumulator)
thread_per_block = 32
blocks = (hblock * wblock + thread_per_block - 1) // thread_per_block
# block_maxima = np.zeros((hblock, wblock), dtype=[('x', int), ('y', int), ('hits', int)])
start = time.time()
accumulate[blocks, thread_per_block](edge_pixel_src, num_edge_pixels_src, orientation_src, width_src, height_src, accumulator, r_table_1D)
end = time.time()
accumulator = accumulator.copy_to_host()

maxima_threshold = THRESHOLD_RATIO * accumulator.max()
time_process += end - start
#
# # Draw
plt.imshow(src)
for j in range(hblock):
    for i in range(wblock):
        if accumulator[j][i] > maxima_threshold:
            plt.plot([accumulator[j, i, 0] * BLOCK_SIZE + BLOCK_SIZE // 2], [accumulator[j, i, 0] * BLOCK_SIZE + BLOCK_SIZE // 2], marker='o', color="yellow")

plt.savefig(f'{IMAGE_DIR}/output.png')
plt.show()

print("----------End accumulating src----------\n")
print(f"Time process: {time_process}s\n")