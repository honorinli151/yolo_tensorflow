# @Author: chenleli
# @Date:   2018-7-12 15:04:31
# @Email:  chenle.li@student.ecp.fr
# @Filename: tf_record.py
# @Last modified by:   chenleli
# @Last modified time: 2018-7-12 22:55:28


import os
import tensorflow as tf
import numpy


# test file
def test_tfrecord();
    for example in tf.python_io.tf_record_iterator("/home/LAB/fusd/.kaggle/competitions/oid/train_tfrecord/train.tfrecord-00799"):
        result = tf.train.Example.FromString(example)
        print(result)
    #     print(result.features.feature['image/class/label'])
        break


def get_features():
    # output file name string to a queue
    filename_queue = tf.train.string_input_producer(['test.tfrecord'], num_epochs=None)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example,
            features={
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/class/text': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
                'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
                'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
                'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
                'image/object/class/label': tf.FixedLenFeature([], tf.int64),
                'image/object/class/text': tf.FixedLenFeature([], tf.string),
                'image/object/depiction': tf.FixedLenFeature([], tf.int64),
                'image/object/group_of': tf.FixedLenFeature([], tf.int64),
                'image/object/occluded': tf.FixedLenFeature([], tf.int64),
                'image/object/truncated': tf.FixedLenFeature([], tf.int64),
                'image/source_id': tf.FixedLenFeature([], tf.string)
            }
        )
    feature_names = features.keys()

a_batch, b_batch, c_batch = tf.train.shuffle_batch([a_out, b_out, c_out], batch_size=3,
                                capacity=200, min_after_dequeue=100, num_threads=2)


class tf_record():
    def _init_(self, phase):
        self.phase = phase
        self.phase_path = phase + 'tfrecord'
        self.data_path = os.path.join(cfg.TF_Record, self.phase_path)
        self
