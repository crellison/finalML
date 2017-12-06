from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from csv import reader

DATA_DIR = 'data/train/'
LABELS = 'labels.csv'

def main():
  datagen = ImageDataGenerator(
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')

def data_reader():
  with open(LABELS) as csvfile:
    filereader = reader(csvfile)

if __name__ == '__main__':
  main()