import numpy as np

# numpy array sobel filter
sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def keepPixel(magnitude, i, j, gradient):
    neighbourOnei = i
    neighbourOnej = j
    neighbourTwoi = i
    neighbourTwoj = j

    if gradient == 0:
        neighbourOnei -= 1
        neighbourTwoi += 1
    elif gradient == 45:
        neighbourOnej -= 1
        neighbourOnei += 1
        neighbourTwoj += 1
        neighbourTwoi -= 1
    elif gradient == 90:
        neighbourOnej -= 1
        neighbourTwoj += 1
    else:  # gradient == 135
        neighbourOnej -= 1
        neighbourOnei -= 1
        neighbourTwoj += 1
        neighbourTwoi += 1

    neighbourOne = magnitude.data[neighbourOnej * magnitude.width + neighbourOnei] if (
            0 <= neighbourOnei < magnitude.width and 0 <= neighbourOnej < magnitude.height) else 0
    neighbourTwo = magnitude.data[neighbourTwoj * magnitude.width + neighbourTwoi] if (
            0 <= neighbourTwoi < magnitude.width and 0 <= neighbourTwoj < magnitude.height) else 0
    cur = magnitude.data[j * magnitude.width + i]

    return (neighbourOne <= cur) and (neighbourTwo <= cur)


# Keep Pixel function by numpy
def keep_pixel_np(magnitude, pixel_gradient):
    height, width = magnitude.data.shape
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

    neighbour_one = np.zeros_like(magnitude.data).astype(int)
    neighbour_one[valid_neighbour_one] = magnitude.data[
        neighbour_one_j[valid_neighbour_one], neighbour_one_i[valid_neighbour_one]]
    neighbour_two = np.zeros_like(magnitude.data).astype(int)
    neighbour_two[valid_neighbour_two] = magnitude.data[
        neighbour_two_j[valid_neighbour_two], neighbour_two_i[valid_neighbour_two]]
    cur = magnitude.data

    return (neighbour_one <= cur) & (neighbour_two <= cur)
