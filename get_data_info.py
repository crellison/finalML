from PIL import Image
import os
import numpy

IMAGE_PATH = os.path.join('data', 'train')

def get_median_size():
    widths = []
    heights = []

    print(os.path.abspath(IMAGE_PATH))

    for image_file in os.listdir(IMAGE_PATH):
        with Image.open(os.path.join(IMAGE_PATH, image_file)) as image:
            widths.append(image.width)
            heights.append(image.width)

    widths.sort()
    heights.sort()

    median_size = (widths[len(widths) // 2], heights[len(heights) // 2])
    return widths, heights, median_size

def main():
    widths, heights, median_size = get_median_size()
    print('Median image size: ', median_size)
    # OUTPUTS (800, 800)
    
    # width_histogram = numpy.histogram(widths)
    # height_histogram = numpy.histogram(heights)
    # print(width_histogram)
    # print(height_histogram)
    
if __name__ == '__main__':
    main()


