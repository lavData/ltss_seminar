import math

import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.image import Image, GrayImage
from src.utils import keep_pixel_np, sobel_filter_x, sobel_filter_y

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


class SeqGeneralHoughTransform:
    def __init__(self, src, template):
        self.src = src
        self.template = template
        self.r_table = [[] for _ in range(N_ROTATION_SLICES)]

    def process_template(self):
        print("----------Start processing template----------\n")

        # Gray convert
        gray_src = GrayImage(template.data.shape[1], template.data.shape[0])
        self.convertToGray(template, gray_src)

        # Sobel filter
        magnitude_x = GrayImage(template.data.shape[1], template.data.shape[0])
        magnitude_y = GrayImage(template.data.shape[1], template.data.shape[0])
        self.convolve(sobel_filter_x, gray_src, magnitude_x, 'x')
        self.convolve(sobel_filter_y, gray_src, magnitude_y, 'y')

        # Magnitude and orientation
        magnitude = GrayImage(template.data.shape[1], template.data.shape[0])
        self.magnitude(magnitude_x, magnitude_y, magnitude)
        orientation = GrayImage(template.data.shape[1], template.data.shape[0])
        self.orientation(magnitude_x, magnitude_y, orientation)

        # Edge minmax
        edge_minmax = GrayImage(template.data.shape[1], template.data.shape[0])
        self.edgemns(magnitude, orientation, edge_minmax)

        # Threshold
        mag_threshold = GrayImage(template.data.shape[1], template.data.shape[0])
        self.threshold(edge_minmax, mag_threshold, THRESHOLD)

        # Create R-table
        self.create_r_table(orientation, mag_threshold)


    def convertToGray(self, image, result):
        result.data = np.mean(image.data, axis=2)
        # plt.imshow(result.data, cmap='gray')
        # plt.savefig(f'{IMAGE_DIR}/gray_src.png')
        # plt.show()

    def convolve(self, sobel_filter: numpy.array, gray_src: GrayImage, result: GrayImage, axis='x'):
        """
        :param sobel_filter:
        :param gray_src:
        :param result:
        :param axis: 'x' or 'y'
        :return:
        """
        if sobel_filter.shape != (3, 3):
            raise Exception("Sobel filter must be 3x3")

        result.data = np.convolve(gray_src.data.flatten(), sobel_filter.flatten(), 'same').reshape(gray_src.data.shape)
        # plt.imshow(result.data, cmap='gray')
        # plt.savefig(f'{IMAGE_DIR}/convolve_{axis}.png')
        # plt.show()

    def magnitude(self, magnitude_x: GrayImage, magnitude_y: GrayImage, result: GrayImage):
        result.data = np.sqrt(np.square(magnitude_x.data) + np.square(magnitude_y.data))
        # plt.imshow(result.data, cmap='gray')
        # plt.savefig(f'{IMAGE_DIR}/magnitude.png')
        # plt.show()

    def orientation(self, gradient_x, gradient_y, result):
        phi = np.arctan2(gradient_y.data, gradient_x.data)
        result.data = np.mod(phi * 180 / np.pi + 360, 360)

    def edgemns(self, magnitude: GrayImage, orientation: GrayImage, result):
        pixel_gradient = ((orientation.data // 45).astype(int) * 45 % 180)
        result.data = np.where(keep_pixel_np(magnitude, pixel_gradient), magnitude.data, 0)
        # plt.imshow(result.data, cmap='gray')
        # plt.savefig(f'{IMAGE_DIR}/minmax.png')
        # plt.show()

    def threshold(self, magnitude: GrayImage, result: GrayImage, threshold: int):
        result.data = np.where(magnitude.data > threshold, 255, 0)
        # plt.imshow(result.data, cmap='gray')
        # plt.savefig(f'{IMAGE_DIR}/threshold.png')
        # plt.show()

    def create_r_table(self, orientation, magnitude_threshold):
        indices_j, indices_i = np.where(magnitude_threshold.data == 255)

        phi = np.fmod(orientation.data[indices_j, indices_i], 360)
        i_slice = (phi / DELTA_ROTATION_ANGLE).astype(int)

        center_x = orientation.width // 2
        center_y = orientation.height // 2
        entry_x = center_x - indices_i
        entry_y = center_y - indices_j

        r = np.sqrt(entry_x ** 2 + entry_y ** 2)
        alpha = np.arctan2(entry_y, entry_x)

        for i in range(len(indices_i)):
            entry = {'r': r[i], 'alpha': alpha[i]}
            self.r_table[i_slice[i]].append(entry)

    def accumulate(self, mag_threshold: GrayImage, orient: GrayImage):
        width = self.src.data.shape[1]
        height = self.src.data.shape[0]
        wblock = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
        hblock = (height + BLOCK_SIZE - 1) // BLOCK_SIZE

        accumulator = np.zeros((hblock, wblock), dtype=np.int32)
        block_maxima = np.zeros((hblock, wblock), dtype=[('x', int), ('y', int), ('hits', int)])

        _max = 0
        for j in range(height):
            for i in range(width):
                if mag_threshold.data[j][i] == 255:
                    phi = orient.data[j][i]
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

        maxima_thres = round(_max * THRESHOLD_RATIO)
        figure = plt.figure(figsize=(12, 12))
        subplot1 = figure.add_subplot(1, 4, 1)
        subplot1.imshow(self.src.data)
        subplot2 = figure.add_subplot(1, 4, 2)
        subplot2.imshow(mag_threshold.data, cmap="gray")
        for j in range(hblock):
            for i in range(wblock):
                if block_maxima[j][i]['hits'] > maxima_thres:
                    subplot2.plot([block_maxima[j][i]['x']], [block_maxima[j][i]['y']], marker='o', color="yellow")

        plt.savefig(f'{IMAGE_DIR}/output.png')
        plt.show()

    def accumulate_src(self):
        print("----------Start accumulating src----------\n")
        # Gray convert
        gray_src = GrayImage(self.src.data.shape[1], self.src.data.shape[0])
        self.convertToGray(self.src, gray_src)

        # Sobel filter
        magnitude_x = GrayImage(self.src.data.shape[1], self.src.data.shape[0])
        magnitude_y = GrayImage(self.src.data.shape[1], self.src.data.shape[0])
        self.convolve(sobel_filter_x, gray_src, magnitude_x, 'x')
        self.convolve(sobel_filter_y, gray_src, magnitude_y, 'y')

        # Magnitude and orientation
        magnitude = GrayImage(self.src.data.shape[1], self.src.data.shape[0])
        self.magnitude(magnitude_x, magnitude_y, magnitude)
        orientation = GrayImage(self.src.data.shape[1], self.src.data.shape[0])
        self.orientation(magnitude_x, magnitude_y, orientation)

        # Edge minmax
        edge_minmax = GrayImage(self.src.data.shape[1], self.src.data.shape[0])
        self.edgemns(magnitude, orientation, edge_minmax)

        # Threshold
        mag_threshold = GrayImage(self.src.data.shape[1], self.src.data.shape[0])
        self.threshold(edge_minmax, mag_threshold, THRESHOLD)

        # Accumulate
        self.accumulate(mag_threshold, orientation)
        print("----------End accumulating src----------\n")


if __name__ == "__main__":
    template = cv2.imread("../../images/leaf.png")
    src = cv2.imread("../../images/leaves.png")
    template = Image(template.shape[1], template.shape[0], template)
    src = Image(src.shape[1], src.shape[0], src)
    a = SeqGeneralHoughTransform(src, template)
    a.process_template()
    a.accumulate_src()