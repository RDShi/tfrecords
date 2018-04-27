from threading import Thread
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import time


def _int64list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _byteslist(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def make_example(list_file,t):
    path = "./data"
    writer = tf.python_io.TFRecordWriter("./tfrecord/"+list_file[0][:-4])
    for file in list_file:
        image = Image.open(os.path.join(path, file))
        image = np.array(image)
        example = tf.train.Example(features=tf.train.Features(feature={"shape": _int64list(image.shape),
                                                                       "image": _byteslist([bytes(image)])}))
        writer.write(example.SerializeToString())
    writer.close()
    print(list_file[0][:-4],time.time() - t)


if __name__ == "__main__":
    t = time.time()
    path = "./data"
    list_file = os.listdir(path)
    n = 500
    for i in range(0, len(list_file), n):
        p = Thread(target=make_example, args=(list_file[i:i+n],t))
        p.start()

    print(time.time()-t)
