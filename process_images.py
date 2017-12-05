from PIL import Image
import os
from sys import argv


def resize_image(image, width):
    return image.resize((width, width))


def square_image(image):
    # TODO: Change to crop from the center
    smallest_dimension = min(image.size)
    return image.crop((0, 0, smallest_dimension, smallest_dimension))


def save_image(image, path):
    with open(os.path.abspath(path), 'w+') as file:
        image.save(file, "JPEG")


def print_usage():
    print('====== Usage ======')
    print('python3 resize_images.py <directory> <scale>')
    print()
    print('directory <str> path of dir with images to process')
    print('scale <int> new width of images')


def main(directory, width):
    print(directory)
    files = os.listdir(directory)
    files.sort()
    small_directory = os.listdir(os.path.join('data', 'small', 'train'))
    print(files)

    if '54305.jpg' in small_directory:
        print('YES IT IS THERE')

    for image_file in files:
        if image_file not in small_directory:
            print('file name:', image_file)
            with Image.open(os.path.join(directory, image_file)) as image:
                # image.show()
                squared = square_image(image)
                new_image = resize_image(squared, width)
                save_path = os.path.join('data', 'small', 'train', image_file)
                # new_image.show()
                try:
                    save_image(new_image, save_path)
                except:
                    print('uh oh error, image:', image_file)
            print(image_file)



if __name__ == '__main__':

    main(os.path.join('data', 'train'), 100)
    """
    if len(argv) < 4:
        print_usage()
    else:
        directory = path.join(argv[2])
        scale = int(argv[3])
        main(directory, scale)
    """