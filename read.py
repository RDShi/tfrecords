import tensorflow as tf
from PIL import Image

path = "./image"
reader = tf.TFRecordReader()
file_queue = tf.train.string_input_producer(["./tfrecord/000005"])

_, serialize_example = reader.read(file_queue)
features = tf.parse_single_example(serialize_example, features={"shape":tf.FixedLenFeature([],tf.int64),
                                                                "image":tf.FixedLenFeature([],tf.string)})

shape = features["shape"]
image = tf.decode_raw(features["image"], tf.uint8)
# image = image.reshape(shape[0], shape[1],3)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

im = sess.run(image)

sess.close()
im = im.reshape(500, 375, 3)
im = Image.fromarray(im)

im.show()


