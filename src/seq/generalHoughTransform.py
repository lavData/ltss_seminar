import math
import time
import numpy as np
import matplotlib.pyplot as plt


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


# Keep Pixel function by numpy
def keep_pixel_np(magnitude: np.ndarray, pixel_gradient: np.ndarray):
    height, width = magnitude.shape
    neighbour_one_i = np.tile(np.arange(width), (height, 1)) - (pixel_gradient == 0) + (pixel_gradient == 45) - (
            pixel_gradient == 135)
    neighbour_one_j = np.tile(np.arange(height)[:, np.newaxis], width) - (pixel_gradient == 45) - (
            pixel_gradient == 90) - (pixel_gradient == 135)
    neighbour_two_i = np.tile(np.arange(width), (height, 1)) + (pixel_gradient == 0) - (pixel_gradient == 45) + (
            pixel_gradient == 135)
    neighbour_two_j = np.tile(np.arange(height)[:, np.newaxis], width) + (pixel_gradient == 45) + (
            pixel_gradient == 90) + (pixel_gradient == 135)

    valid_neighbour_one = (neighbour_one_i >= 0) & (neighbour_one_i < width) & (neighbour_one_j >= 0) & (
            neighbour_one_j < height)
    valid_neighbour_two = (neighbour_two_i >= 0) & (neighbour_two_i < width) & (neighbour_two_j >= 0) & (
            neighbour_two_j < height)

    neighbour_one = np.zeros_like(magnitude).astype(int)
    neighbour_one[valid_neighbour_one] = magnitude[
        neighbour_one_j[valid_neighbour_one], neighbour_one_i[valid_neighbour_one]]
    neighbour_two = np.zeros_like(magnitude).astype(int)
    neighbour_two[valid_neighbour_two] = magnitude[
        neighbour_two_j[valid_neighbour_two], neighbour_two_i[valid_neighbour_two]]
    cur = magnitude

    return (neighbour_one <= cur) & (neighbour_two <= cur)


class SeqGeneralHoughTransform:
    def __init__(self, src: np.array, template: np.array, image_dir=IMAGE_DIR):
        self.src = src
        self.height_src = src.shape[0]
        self.width_src = src.shape[1]
        self.template = template
        self.height_template = template.shape[0]
        self.width_template = template.shape[1]
        self.r_table = [[] for _ in range(N_ROTATION_SLICES)]
        self.image_dir = image_dir

    def process_template(self):
        print("----------Start processing template----------\n")
        time_process = 0

        # Gray convert
        start = time.time()
        gray_template = self.convertToGray(self.template)
        end = time.time()
        time_process += end - start

        # Sobel filter
        start = time.time()
        magnitude_x = self.convolve(sobel_filter_x, gray_template)
        magnitude_y = self.convolve(sobel_filter_y, gray_template)
        end = time.time()
        time_process += end - start

        # Magnitude and orientation
        start = time.time()
        magnitude = self.magnitude(magnitude_x, magnitude_y)
        orientation = self.orientation(magnitude_x, magnitude_y)
        end = time.time()
        time_process += end - start

        # Edge minmax
        start = time.time()
        edge_minmax = self.edgemns(magnitude, orientation)
        end = time.time()
        time_process += end - start

        # Threshold
        start = time.time()
        mag_threshold = self.threshold(edge_minmax, THRESHOLD, type_input='template')
        end = time.time()
        time_process += end - start

        # Create R-table
        start = time.time()
        self.create_r_table(orientation, mag_threshold)
        end = time.time()
        time_process += end - start

        print("----------End processing template----------\n")
        print(f"Time processing template: {time_process}\n")

    def accumulate_src(self):
        print("----------Start accumulating src----------\n")
        time_process = 0

        # Gray convert
        start = time.time()
        gray_src = self.convertToGray(self.src)
        end = time.time()
        time_process += end - start

        # Sobel filter
        start = time.time()
        magnitude_x = self.convolve(sobel_filter_x, gray_src)
        magnitude_y = self.convolve(sobel_filter_y, gray_src)
        end = time.time()
        time_process += end - start

        # Magnitude and orientation
        start = time.time()
        magnitude = self.magnitude(magnitude_x, magnitude_y)
        orientation = self.orientation(magnitude_x, magnitude_y)
        end = time.time()
        time_process += end - start

        # Edge minmax
        start = time.time()
        edge_minmax = self.edgemns(magnitude, orientation)
        end = time.time()
        time_process += end - start

        # Threshold
        start = time.time()
        mag_threshold = self.threshold(edge_minmax, THRESHOLD)
        end = time.time()
        time_process += end - start

        # Accumulate
        start = time.time()
        self.accumulate(mag_threshold, orientation)
        end = time.time()
        time_process += end - start

        print("----------End accumulating src----------\n")
        print(f"Time process: {time_process}s\n")

    def convertToGray(self, image, type_input=None):
        result = np.mean(image.data, axis=2)
        if type_input in ['template', 'src']:
            plt.imshow(result, cmap='gray')
            plt.savefig(f'{self.image_dir}/gray_{type_input}.png')
        return result

    def convolve(self, sobel_filter: np.array, gray_src: np.array, axis='x', type_input=None):
        if sobel_filter.shape != (3, 3):
            raise Exception("Sobel filter must be 3x3")

        result = np.convolve(gray_src.flatten(), sobel_filter.flatten(), 'same').reshape(gray_src.shape)
        if type_input in ['template', 'src']:
            plt.imshow(result, cmap='gray')
            plt.savefig(f'{self.image_dir}/sobel_{axis}_{type_input}.png')
        return result

    def magnitude(self, magnitude_x: np.array, magnitude_y: np.array, type_input=None):
        result = np.sqrt(np.square(magnitude_x) + np.square(magnitude_y))
        if type_input in ['template', 'src']:
            plt.imshow(result, cmap='gray')
            plt.savefig(f'{self.image_dir}/magnitude_{type_input}.png')
        return result

    def orientation(self, magnitude_x: np.array, magnitude_y: np.array):
        phi = np.arctan2(magnitude_y, magnitude_x)
        result = np.mod(phi * 180 / np.pi + 360, 360)
        return result

    def edgemns(self, magnitude: np.array, orientation: np.array, type_input=None):
        pixel_gradient = ((orientation // 45).astype(int) * 45 % 180)
        result = np.where(keep_pixel_np(magnitude, pixel_gradient), magnitude, 0)
        if type_input in ['template', 'src']:
            plt.imshow(result, cmap='gray')
            plt.savefig(f'{self.image_dir}/edge_minmax_{type_input}.png')
        return result

    def threshold(self, magnitude: np.array, threshold: int, type_input=None):
        result = np.where(magnitude > threshold, 255, 0)
        if type_input in ['template', 'src']:
            plt.imshow(result, cmap='gray')
            plt.savefig(f'{self.image_dir}/threshold_{type_input}.png')
            plt.plot()
        return result

    def create_r_table(self, orientation: np.array, magnitude_threshold: np.array):
        indices_j, indices_i = np.where(magnitude_threshold == 255)

        phi = np.fmod(orientation[indices_j, indices_i], 360)
        i_slice = (phi / DELTA_ROTATION_ANGLE).astype(int)

        center_x = self.width_template // 2
        center_y = self.height_template // 2
        entry_x = center_x - indices_i
        entry_y = center_y - indices_j

        r = np.sqrt(entry_x ** 2 + entry_y ** 2)
        alpha = np.arctan2(entry_y, entry_x)

        for i in range(len(indices_i)):
            entry = {'r': r[i], 'alpha': alpha[i]}
            self.r_table[i_slice[i]].append(entry)

    def accumulate(self, mag_threshold: np.array, orient: np.array):
        width = self.width_src
        height = self.height_src
        wblock = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
        hblock = (height + BLOCK_SIZE - 1) // BLOCK_SIZE

        accumulator = np.zeros((hblock, wblock), dtype=np.int32)
        block_maxima = np.zeros((hblock, wblock), dtype=[('x', int), ('y', int), ('hits', int)])

        _max = 0
        for j in range(height):
            for i in range(width):
                if mag_threshold[j][i] == 255:
                    phi = orient[j][i]
                    i_slice = int(phi // DELTA_ROTATION_ANGLE)
                    entries = self.r_table[i_slice]
                    for entry in entries:
                        r = entry['r']
                        alpha = entry['alpha']
                        xc = int(i + r * math.cos(alpha))
                        yc = int(j + r * math.sin(alpha))

                        if xc < 0 or xc >= width or yc < 0 or yc >= height:
                            continue
                        accumulator[yc // BLOCK_SIZE][xc // BLOCK_SIZE] += 1
                        block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['hits'] = accumulator[yc // BLOCK_SIZE][
                            xc // BLOCK_SIZE]
                        block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['x'] = xc
                        block_maxima[yc // BLOCK_SIZE][xc // BLOCK_SIZE]['y'] = yc
                        if accumulator[yc // BLOCK_SIZE][xc // BLOCK_SIZE] > _max:
                            _max = accumulator[yc // BLOCK_SIZE][xc // BLOCK_SIZE]

        maxima_threshold = round(_max * THRESHOLD_RATIO)
        plt.imshow(self.src)
        for j in range(hblock):
            for i in range(wblock):
                if block_maxima[j][i]['hits'] > maxima_threshold:
                    plt.plot([block_maxima[j][i]['x']], [block_maxima[j][i]['y']], marker='o', color="yellow")

        plt.savefig(f'{self.image_dir}/output.png')
        plt.show()
