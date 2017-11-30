from PIL import Image
import os
import numpy

image_path = os.path.join('data', 'train')

widths = []
heights = []

print(os.path.abspath(image_path))

for image_file in os.listdir(image_path):
    with Image.open(os.path.join(image_path, image_file)) as image:
        widths.append(image.width)
        heights.append(image.width)

widths.sort()
heights.sort()

median_size = (widths[len(widths) // 2], heights[len(heights) // 2])
print('Median image size: ', median_size)

"""
width_histogram = numpy.histogram(widths)
height_histogram = numpy.histogram(heights)
print(width_histogram)
print(height_histogram)
"""
