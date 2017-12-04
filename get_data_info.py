from PIL import Image
import os
import numpy
from collections import Counter

IMAGE_PATH = os.path.join('data', 'train')

def get_size_info():
    widths = Counter()
    heights = Counter()

    print(os.path.abspath(IMAGE_PATH))

    for image_file in os.listdir(IMAGE_PATH):
        with Image.open(os.path.join(IMAGE_PATH, image_file)) as image:
            widths[image.width] += 1
            heights[image.height] += 1

    widths.sort()
    heights.sort()

    median_size = (widths[len(widths) // 2], heights[len(heights) // 2])
    return widths, heights, median_size

def main():
    widths, heights, median_size = get_size_info()
    print('Median image size: ', median_size)
    # OUTPUTS (800, 800)
    print(widths)
    print(heights)


    
if __name__ == '__main__':
    main()


