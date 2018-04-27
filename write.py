import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time


def _int64list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _byteslist(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

t = time.time()
path = "./data"
list_file = os.listdir(path)
writer = tf.python_io.TFRecordWriter("test.tfr")
for file in list_file:
    image = Image.open(os.path.join(path, file))
    image = np.array(image)
    example = tf.train.Example(features=tf.train.Features(feature={"shape":_int64list(image.shape),
                                                                   "image":_byteslist([bytes(image)])}))
    writer.write(example.SerializeToString())
writer.close()
print(time.time() - t)


