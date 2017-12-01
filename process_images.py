from PIL import Image
from os import path, listdir
from sys import argv

def resize_image(image, width):
  pass

def square_image(image):
  pass

def save_image(image, path):
  pass

def print_usage():
  print('====== Usage ======')
  print('python3 resize_images.py <directory> <scale>')
  print()
  print('directory <str> path of dir with images to process')
  print('scale <int> new width of images')

def main(directory, scale):
  for image_file in listdir(directory):
    with Image.open(path.join(directory, image_file)) as image:
      squared = square_image(image)
      new_image = resize_image(squared)
      save_path = 'placeholder'
      save_image(new_image, save_path)

if __name__ == '__main__':
  if len(argv) < 4:
    print_usage()
  else:
    directory = path.join(argv[2])
    scale = int(argv[3])
    main(directory, scale)