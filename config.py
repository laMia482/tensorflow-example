import logging
logging.basicConfig(level = logging.INFO)
def puts_debug(x):
  logging.debug(x)
def puts_info(x):
  logging.info(x)

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_boolean('is_train', True, 'train the model')
flags.DEFINE_boolean('is_eval', True, 'eval the model')
flags.DEFINE_integer('image_width', 64, 'the width of the input image')
flags.DEFINE_integer('image_height', 64, 'the height of the input image')
flags.DEFINE_integer('image_channel', 3, 'the channel of the input image')
flags.DEFINE_integer('num_classes', 2, 'the category of the predictions')
flags.DEFINE_integer('max_predictions', 5, 'the locatation of the predictions')
flags.DEFINE_integer('batch_size', 320, 'batch size when running network')
flags.DEFINE_integer('epoch', 1000, 'epoch of training the all data')
flags.DEFINE_integer('test_iter', 10, 'iteration when training')
flags.DEFINE_integer('save_iter', 100, 'iteration when training')
flags.DEFINE_float('init_learning_rate', 0.001, 'the initial value of learning rate')
flags.DEFINE_string('network', 'default_network', 'network arch')
flags.DEFINE_string('train_dataset', 'data/train.json', 'data for training')
flags.DEFINE_string('eval_dataset', 'data/eval.json', 'data for evaluating')
flags.DEFINE_string('save_path', 'ckpt', 'floder to save models')

cfg = flags.FLAGS
