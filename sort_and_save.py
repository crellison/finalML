from wrangledata import get_train_data, SORTED_INFO

def main():
  headers, data = get_train_data()
  # sort by the painterID
  data = sorted(data, key=lambda x: x[1])
  with open(SORTED_INFO, 'w+') as writer:
    for line in data:
      fileline = ','.join(line) + '\n'
      writer.write(fileline)

if __name__ == '__main__':
  main()