import tensorflow as tf
import glob, random


files = glob.glob('res/*.jpg', recursive=True)
random.shuffle(files)
labels = list(map(lambda x:0 if 'cat' in x else 1, files))
print(files)
print(labels)

db = list(zip(files, labels))
split_len = round(len(db) * 0.667)

train_x, train_y = zip(*db[:split_len])
test_x, test_y = zip(*db[split_len:])



print(train_x, train_y)
print(test_x, test_y)