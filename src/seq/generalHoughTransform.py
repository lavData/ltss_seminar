import math

import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.image import Image, GrayImage
from src.utils import keep_pixel_np, sobel_filter_x, sobel_filter_y

THRESHOLD = 200
IMAGE_DIR = '../../images'

class SeqGeneralHoughTransform:
    def __init__(self, src, template):
        self.src = src
        self.template = template

    def process_template(self):
        print("----------Start processing template----------\n")

        gray_src = GrayImage(self.src.width, self.src.height)
        self.convertToGray(self.src, gray_src)

    def convertToGray(self, image, result):
        result.data = np.mean(image.data, axis=2)
        plt.imshow(result.data, cmap='gray')
        plt.savefig(f'{IMAGE_DIR}/gray_src.png')
        plt.show()

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
        plt.imshow(result.data, cmap='gray')
        plt.savefig(f'{IMAGE_DIR}/convolve_{axis}.png')
        plt.show()

    def magnitude(self, magnitude_x: GrayImage, magnitude_y: GrayImage, result: GrayImage):
        result.data = np.sqrt(np.square(magnitude_x.data) + np.square(magnitude_y.data))
        plt.imshow(result.data, cmap='gray')
        plt.savefig(f'{IMAGE_DIR}/magnitude.png')
        plt.show()

    def orientation(self, gradient_x, gradient_y, result):
        phi = np.arctan2(gradient_y.data, gradient_x.data)
        result.data = np.mod(phi * 180 / np.pi + 360, 360)

    def edgemns(self, magnitude: GrayImage, orientation: GrayImage, result):
        pixel_gradient = ((orientation.data // 45).astype(int) * 45 % 180)
        result.data = np.where(keep_pixel_np(magnitude, pixel_gradient), magnitude.data, 0)

    def threshold(self, magnitude, result, threshold):
        for i in range(0, magnitude.width * magnitude.height):
            result.data[i] = 255 if magnitude.data[i] > threshold else 0

    def create_r_table(self, orientation, magnitude_threshold):
        pass

if __name__ == "__main__":
    template = cv2.imread("../../images/lane.png")
    template = Image(template.shape[1], template.shape[0], template)
    a = SeqGeneralHoughTransform(None, template)
    gray_src = GrayImage(template.data.shape[1], template.data.shape[0])
    a.convertToGray(template, gray_src)
    magnitude_x = GrayImage(template.data.shape[1], template.data.shape[0])
    magnitude_y = GrayImage(template.data.shape[1], template.data.shape[0])
    magnitude = GrayImage(template.data.shape[1], template.data.shape[0])
    a.convolve(sobel_filter_x, gray_src, magnitude_x, 'x')
    a.convolve(sobel_filter_y, gray_src, magnitude_y, 'y')
    a.magnitude(magnitude_x, magnitude_y, magnitude)
    orientation = GrayImage(template.data.shape[1], template.data.shape[0])
    a.orientation(magnitude_x, magnitude_y, orientation)
    plt.imshow(magnitude.data, cmap='gray')
    x, y = np.meshgrid(np.arange(orientation.width), np.arange(orientation.height))
    dx = np.cos(np.deg2rad(orientation.data))
    dy = -np.sin(np.deg2rad(orientation.data))
    plt.quiver(x, y, dx, dy, orientation.data, cmap='gray')
    plt.show()