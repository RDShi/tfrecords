from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import freeze_support
import numpy as np
import tensorflow as tf
import os
from PIL import Image

import time

def w1(func):
    def inner(*args,**kwargs):
        past = time.time()
        func(*args,**kwargs)
        now = time.time()
        cost_time = now - past
        print("The function <%s> cost time: <%s>"%(func.func_name,cost_time))
    return inner

def _int64list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _byteslist(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def make_example(list_file):
    path = "./data"
    writer = tf.python_io.TFRecordWriter("./tfrecord/"+list_file[0][:-4])
    for file in [list_file]:
        image = Image.open(os.path.join(path, file))
        image = np.array(image)
        example = tf.train.Example(features=tf.train.Features(feature={"shape": _int64list(image.shape),
                                                                       "image": _byteslist([bytes(image)])}))
        writer.write(example.SerializeToString())
    writer.close()

path = "./data"
list_file = os.listdir(path)

ppool = Pool(4)
@w1
def MulProcess():
    for n in list_file:
        ppool.apply(func=make_example, args=(n,))
    ppool.close()
    ppool.join()
MulProcess()

tpool = ThreadPool(4)
@w1
def MulThreading():
    for n in list_file:
        tpool.apply(func=make_example, args=(n,))
    tpool.close()
    tpool.join()
MulThreading()