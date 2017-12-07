from sklearn.metrics import accuracy_score, confusion_matrix
from random import sample, randint, choice
from PIL import Image

def show_img(img):
  ''' opens image with  builtin viewer '''
  img_ = img * 255
  img_ = Image.fromarray(img_, mode='L')
  img_.show(command='fim')


def make_hash(y_data):
  ''' makes a hastable of  indecies of each class of item '''
  label_map = {x: list() for x in range(10)}
  for i in range(len(y_data)):
    label = y_data[i]
    label_map[label].append(i)
  return label_map


def create_pairs(label_map):
  ''' generates positive and negative pairs from a hashmap from make_hash '''
  smallest_group = min(len(label_map[label]) for label in range(10)) - 1
  pairs = []
  labels = []
  for label in range(10):
    for pair_num in range(smallest_group):
      a, b = sample(label_map[label], 2)

      other_label = randint(0, 9)

      if other_label == label:
        other_label = (other_label + 1) % 10

      c = choice(label_map[label])
      d = choice(label_map[other_label])

      pairs.extend([(a, b), (c, d)])
      labels.extend([1, 0])

  return labels, pairs


def outfile(dataname):
  ''' generates a filename for saving a model with a timehash '''
  signature = floor(time())
  return '%s_model_%i.h5' % (dataname, signature)


def eval_siamese(test_pred, test_labels):
  '''
  prints evaluation of siamese CNN from predictions and ground truth
  '''
  test_pred = test_pred < 0.5
  test_truth = np.array(test_labels)

  print()
  print("accuracy is: " + str(accuracy_score(y_pred, y_truth)))
  print("confusion matrix:")
  print(confusion_matrix(y_pred, y_truth))
