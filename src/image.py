import numpy as np

class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = np.empty(3 * width * height)
    

class GrayImage:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = np.empty(width * height)
        
print(3/5)