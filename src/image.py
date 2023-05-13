import numpy as np


class Image:
    def __init__(self, width, height, data: np.ndarray = None):
        self.width = width
        self.height = height
        self.data = data if data is not None else np.zeros((width, height, 3), dtype=np.uint8)


class GrayImage:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = np.zeros((width, height), dtype=np.uint8)
