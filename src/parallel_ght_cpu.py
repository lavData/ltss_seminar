import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from numba import config
config.THREADING_LAYER = 'omp'

THRESHOLD = 200
N_ROTATION_SLICES = 72
MAX_SCALE = 1.4
MIN_SCALE = 0.6
DELTA_SCALE_RATIO = 0.1
N_SCALE_SLICE = int((MAX_SCALE - MIN_SCALE) // DELTA_SCALE_RATIO + 1)
BLOCK_SIZE = 10
THRESHOLD_RATIO = 0.3
DELTA_ROTATION_ANGLE = 360 / N_ROTATION_SLICES
IMAGE_DIR = '../../images'

# numpy array sobel filter
sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


class ParallelGeneralHoughTransformCPU:
    def __init__(self, src: np.array, template: np.array, image_dir=IMAGE_DIR):
        self.src = src
        self.height_src = src.shape[0]
        self.width_src = src.shape[1]
        self.template = template
        self.height_template = template.shape[0]
        self.width_template = template.shape[1]
        self.r_table = [[] for _ in range(N_ROTATION_SLICES)]
        self.image_dir = image_dir
        self.wblock = (self.width_src + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.hblock = (self.height_src + BLOCK_SIZE - 1) // BLOCK_SIZE


    def process_template(self):
        print("----------Start processing template----------\n")
        time_process = 0

        # Gray convert
        gray_template = np.zeros(self.template.shape[:2], dtype=np.float64)
        start = time.time()
        self.convertToGray(self.template, gray_template)
        end = time.time()
        time_process += end - start

        # Sobel filter
        magnitude_x = np.zeros_like(gray_template)
        magnitude_y = np.zeros_like(gray_template)
        start = time.time()
        self.convolve(sobel_filter_x, gray_template, magnitude_x)
        self.convolve(sobel_filter_y, gray_template, magnitude_y)
        end = time.time()
        time_process += end - start

        # Magnitude and orientation
        magnitude_tpl = np.zeros_like(gray_template)
        orientation_tpl = np.zeros_like(gray_template)
        start = time.time()
        self.magnitude(magnitude_x, magnitude_y, magnitude_tpl)
        self.orientation(magnitude_x, magnitude_y, orientation_tpl)
        end = time.time()
        time_process += end - start

        # Edge minmax
        edge_minmax_tpl = np.zeros_like(gray_template)
        start = time.time()
        self.edgemns(magnitude_tpl, orientation_tpl, edge_minmax_tpl)
        end = time.time()
        time_process += end - start

        # Threshold
        threshold_tpl = np.zeros_like(gray_template)
        start = time.time()
        self.threshold(edge_minmax_tpl, THRESHOLD, threshold_tpl, type_input='template')
        end = time.time()
        time_process += end - start

        # Create R-table
        start = time.time()
        self.create_r_table(orientation_tpl, threshold_tpl)
        end = time.time()
        time_process += end - start

        print("----------End processing template----------\n")
        print(f"Time processing template: {time_process}\n")

    def accumulate_src(self):
        print("----------Start accumulating src----------\n")
        time_process = 0

        # Gray convert
        gray_src = np.zeros(self.src.shape[:2], dtype=np.float64)
        start = time.time()
        self.convertToGray(self.src, gray_src)
        end = time.time()
        time_process += end - start

        # Sobel filter
        magnitude_x = np.zeros_like(gray_src)
        magnitude_y = np.zeros_like(gray_src)
        start = time.time()
        self.convolve(sobel_filter_x, gray_src, magnitude_x)
        self.convolve(sobel_filter_y, gray_src, magnitude_y)
        end = time.time()
        time_process += end - start

        # Magnitude and orientation
        magnitude_src = np.zeros_like(gray_src)
        orientation_src = np.zeros_like(gray_src)
        start = time.time()
        self.magnitude(magnitude_x, magnitude_y, magnitude_src)
        self.orientation(magnitude_x, magnitude_y, orientation_src)
        end = time.time()
        time_process += end - start

        # Edge minmax
        edge_minmax = np.zeros_like(gray_src)
        start = time.time()
        self.edgemns(magnitude_src, orientation_src, edge_minmax)
        end = time.time()
        time_process += end - start

        # Threshold
        mag_threshold = np.zeros_like(gray_src)
        start = time.time()
        self.threshold(edge_minmax, THRESHOLD, mag_threshold)
        end = time.time()
        time_process += end - start

        # Accumulate
        accumulator = np.zeros((self.hblock, self.wblock), dtype=np.int32)
        block_maxima = np.zeros((self.hblock, self.wblock), dtype=[('x', int), ('y', int), ('hits', int)])
        start = time.time()
        block_maxima, maxima_threshold = self.accumulate(mag_threshold, orientation_src, accumulator, block_maxima)
        end = time.time()
        time_process += end - start

        # Draw
        wblock = (self.width_src + BLOCK_SIZE - 1) // BLOCK_SIZE
        hblock = (self.height_src + BLOCK_SIZE - 1) // BLOCK_SIZE
        plt.imshow(self.src)
        for j in range(hblock):
            for i in range(wblock):
                if block_maxima[j][i]['hits'] > maxima_threshold:
                    plt.plot([block_maxima[j][i]['x']], [block_maxima[j][i]['y']], marker='o', color="yellow")

        plt.savefig(f'{self.image_dir}/output.png')
        plt.show()

        print("----------End accumulating src----------\n")
        print(f"Time process: {time_process}s\n")

    @jit(parallel=True, cache=True)
    def convertToGray(self, image: np.array, gray: np.array, type_input=None):
        for i in prange(image.shape[0]):
            for j in prange(image.shape[1]):
                gray[i][j] = 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]

    @jit(parallel=True, cache=True)
    def convolve(self, sobel_filter: np.array, gray_src: np.array, magnitude, axis='x', type_input=None):
        for i in prange(gray_src.shape[0]):
            for j in prange(gray_src.shape[1]):
                temp = 0
                for jj in range(-1, 2):
                    for ii in range(-1, 2):
                        jj_pad = max(0, min(gray_src.shape[1] - 1, j + jj))
                        ii_pad = max(0, min(gray_src.shape[0] - 1, i + ii))
                        temp += gray_src[ii_pad][jj_pad] * sobel_filter[ii + 1][jj + 1]
                magnitude[i][j] = temp

    @jit(parallel=True, cache=True)
    def magnitude(self, magnitude_x: np.array, magnitude_y: np.array, _magnitude: np.array, type_input=None):
        for i in prange(magnitude_x.shape[0]):
            for j in prange(magnitude_x.shape[1]):
                _magnitude[i][j] = math.sqrt(magnitude_x[i][j] ** 2 + magnitude_y[i][j] ** 2)

    @jit(parallel=True, cache=True)
    def orientation(self, magnitude_x: np.array, magnitude_y: np.array, _orientation: np.array):
        for i in prange(magnitude_x.shape[0]):
            for j in prange(magnitude_x.shape[1]):
                _orientation[i][j] = (math.atan2(magnitude_y[i][j], magnitude_x[i][j]) * 180 / math.pi + 360) % 360

    @jit(parallel=True, cache=True)
    def edgemns(self, magnitude: np.array, orientation: np.array, result: np.array, type_input=None):
        for i in prange(magnitude.shape[0]):
            for j in prange(magnitude.shape[1]):
                pixel_gradient = int(orientation[i][j] // 45) * 45 % 180
                neighbour_one_i = i
                neighbour_one_j = j
                neighbour_two_i = i
                neighbour_two_j = j

                if pixel_gradient == 0:
                    neighbour_one_i = i - 1
                    neighbour_two_i = i + 1
                elif pixel_gradient == 45:
                    neighbour_one_i = i + 1
                    neighbour_one_j = j - 1
                    neighbour_two_i = i - 1
                    neighbour_two_j = j + 1
                elif pixel_gradient == 90:
                    neighbour_one_j = j - 1
                    neighbour_two_j = j + 1
                elif pixel_gradient == 135:
                    neighbour_one_i = i - 1
                    neighbour_one_j = j - 1
                    neighbour_two_i = i + 1
                    neighbour_two_j = j + 1

                neighbour_one_i = max(0, min(magnitude.shape[0] - 1, neighbour_one_i))
                neighbour_one_j = max(0, min(magnitude.shape[1] - 1, neighbour_one_j))
                neighbour_two_i = max(0, min(magnitude.shape[0] - 1, neighbour_two_i))
                neighbour_two_j = max(0, min(magnitude.shape[1] - 1, neighbour_two_j))

                neighbour_one = magnitude[neighbour_one_i][neighbour_one_j]
                neighbour_two = magnitude[neighbour_two_i][neighbour_two_j]

                if (neighbour_one <= magnitude[i][j]) & (neighbour_two <= magnitude[i][j]):
                    result[i][j] = magnitude[i][j]
                else:
                    result[i][j] = 0

    @jit(parallel=True, cache=True)
    def threshold(self, magnitude: np.array, threshold: int, result: np.array, type_input=None):
        for i in prange(magnitude.shape[0]):
            for j in prange(magnitude.shape[1]):
                if magnitude[i][j] > threshold:
                    result[i][j] = 255
                else:
                    result[i][j] = 0

    @jit(parallel=True, cache=True)
    def create_r_table(self, orientation: np.array, magnitude_threshold: np.array):
        for i in prange(orientation.shape[0]):
            for j in prange(orientation.shape[1]):
                if magnitude_threshold[i][j] == 255:
                    phi = orientation[i][j] % 360
                    i_slice = int(phi // DELTA_ROTATION_ANGLE)

                    center_x = self.width_template // 2
                    center_y = self.height_template // 2
                    entry_x = center_x - j
                    entry_y = center_y - i

                    r = math.sqrt(entry_x ** 2 + entry_y ** 2)
                    alpha = math.atan2(entry_y, entry_x)

                    entry = {'r': r, 'alpha': alpha}
                    self.r_table[i_slice].append(entry)

    @jit(cache=True)
    def accumulate4D(self, mag_threshold: np.array, orient: np.array, accumulator: np.array, block_maxima: np.array,):
        _max = 0
        for j in range(self.height_src):
            for i in range(self.width_src):
                if mag_threshold[j][i] == 255:
                    phi = orient[j][i]
                    for i_theta in range(N_ROTATION_SLICES):
                        theta = i_theta * DELTA_ROTATION_ANGLE
                        theta_r = math.radians(theta)
                        i_slice = int(((phi - theta + 360) % 360) // DELTA_ROTATION_ANGLE)
                        entries = self.r_table[i_slice]
                        for entry in entries:
                            r = entry['r']
                            alpha = entry['alpha']
                            for scale in range(N_SCALE_SLICE):
                                s = scale * DELTA_SCALE_RATIO + MIN_SCALE
                                xc = int(i + r * s * math.cos(alpha + theta_r))
                                yc = int(j + r * s * math.sin(alpha + theta_r))

                                if xc < 0 or xc >= self.width_src or yc < 0 or yc >= self.height_src:
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
    def accumulate(self, mag_threshold: np.array, orient: np.array, accumulator: np.array, block_maxima: np.array):
        _max = 0
        for j in range(self.height_src):
            for i in range(self.width_src):
                if mag_threshold[j][i] == 255:
                    phi = orient[j][i]
                    i_slice = int(phi // DELTA_ROTATION_ANGLE)
                    entries = self.r_table[i_slice]
                    for entry in entries:
                        r = entry['r']
                        alpha = entry['alpha']
                        xc = int(i + r * math.cos(alpha))
                        yc = int(j + r * math.sin(alpha))

                        if xc < 0 or xc >= self.width_src or yc < 0 or yc >= self.height_src:
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
