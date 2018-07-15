# @Author: chenleli
# @Date:   2018-7-12 15:04:31
# @Email:  chenle.li@student.ecp.fr
# @Filename: tf_record.py
# @Last modified by:   chenleli
# @Last modified time: 2018-7-14 17:32:05

"""
Read from tfrecord files, return a queue for training.
"""

import os
import tensorflow as tf
import numpy as np
import pickle
import yolo.config as cfg


def get_train_features():

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


# a_batch, b_batch, c_batch = tf.train.shuffle_batch([a_out, b_out, c_out], batch_size=3,
#                                 capacity=200, min_after_dequeue=100, num_threads=2)


class tf_record():

    def _init_(self, phase):

        self.phase = phase
        self.phase_path = phase + 'tfrecord'
        self.data_path = os.path.join(cfg.TF_Record, self.phase_path)
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.test_tfrecord_path = os.listdir(self.data_path)[0]

    def input_pipeline_queque():

        # By default, we use the config for train_tfrecord
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        features = tf.parse_example(serialized_example,
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/class/label': tf.VarLenFeature(tf.int64),
            'image/object/class/text': tf.VarLenFeature(tf.string),
            'image/object/depiction': tf.VarLenFeature(tf.int64),
            'image/object/group_of': tf.VarLenFeature(tf.int64),
            'image/object/occluded': tf.VarLenFeature(tf.int64),
            'image/object/truncated': tf.VarLenFeature(tf.int64),
            'image/source_id': tf.FixedLenFeature([], tf.string)
        })
        image, label = features['image/encoded'], features['image/class/label']
        image_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,\
        min_after_dequeue=min_after_dequeue)
        return image_batch, label_batch
        # TODO Check repeat.

    def inspect_tfrecord():

        # inspect given tfrecord file using the first tfrecord in data_dir
        # for example in tf.python_io.tf_record_iterator("/home/LAB/fusd/.kaggle/competitions/oid/train_tfrecord/train.tfrecord-00799"):
        for example in tf.python_io.tf_record_iterator(self.test_tfrecord_path):
            result = tf.train.Example.FromString(example)
            print(result)
        #     print(result.features.feature['image/class/label'])
            break
