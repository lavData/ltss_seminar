from image import GrayImage, Image
import math

class SeqGeneralHoughTransform:
    def __init__(self, src, template):
        self.src = src
        self.template = template
        
    def accumulateSource(self):
        print("----------Start processing and accumulating source----------\n")
        
        graySrc = GrayImage(self.src.width, self.src.height)
        self.convertToGray(self.src, graySrc)
        
        
        
    def convertToGray(self, image, result):
        for i in range(0, image.width * image.height):
            result.data[i] = (image.data[3 * i] + image.data[3 * i + 1] + image.data[3 * i + 2]) / 3
    
    def convolve(self, sobelFilter, graySrc, result):
        if len(sobelFilter) != 3 or len(sobelFilter[0]) != 3:
            print("ERROR: Only apply for 3x3 filter ")
            return
        
        for j in range(0, graySrc.height):
            for i in range(0, graySrc.width):
                tmp = 0.0
                for jj in range(-1,2):
                    for ii in range(-1,2):
                        row = j + jj
                        col = i + ii
                        
                        if row < 0 or row >= graySrc.height or col < 0 or col >= graySrc.width:
                            # out of image bound, do nothing
                            pass
                        else:
                            idx = row * graySrc.width + col
                            tmp += graySrc.data[idx] * sobelFilter[jj + 1][ii + 1]
                
                # do not consider image boundary
                if j == 0 or j == graySrc.height-1 or i == 0 or i == graySrc.width-1:
                    result.data[j * graySrc.width + i] = 0
                else:
                    result.data[j * graySrc.width + i] = tmp
                    
    def magnitude(self, gradientX, gradientY, result):
        for i in range(0, gradientX.width * gradientX.height):
            result.data[i] = math.sqrt(gradientX.data[i] ** 2 + gradientY.data[i] ** 2)
    
    def orientation(self, gradientX, gradientY, result):
        for i in range(0, gradientX.width * gradientX.height):
            result.data[i] = math.fmod(math.atan2(gradientY.data[i], gradientX.data[i]) * 180 / math.pi + 360, 360)
           
        
        
    
    