import numpy as np
import tensorflow as tf
from config import cfg, puts_debug, puts_info

def calc_loss(logits = None, predictions = None):
  if None in [logits, predictions]:
    raise ValueError('None should never arise in calc_loss')
  puts_debug('loss >> {}'.format(predictions[:, :, :cfg.num_classes]))
  puts_debug('loss >> {}'.format(make_onehot(logits[:, :, 0], num_classes = cfg.num_classes)))
  loss_labels = tf.losses.softmax_cross_entropy(onehot_labels = predictions[:, :, :cfg.num_classes],
                                                logits = make_onehot(logits[:, :, 0], num_classes = cfg.num_classes))
  loss_bboxes = tf.losses.mean_squared_error(labels = logits[:, :, 1:], predictions = predictions[:, :, cfg.num_classes:])
  return (loss_labels + loss_bboxes)

def calc_accuracy(logits = None, predictions = None):
  if None in [logits, predictions]:
    raise ValueError('None should never arise in calc_accuracy')
  compare = tf.equal(tf.argmax(predictions, 2), tf.cast(logits[:, :, 0], tf.int64))
  return tf.reduce_mean(tf.cast(compare, tf.float32))
  val = 0
  puts_info('shape: {}, shape: {}'.format(logits.get_shape(), predictions.get_shape()))
  for row in range(cfg.batch_size):
    for col in range(cfg.max_predicitons):
      if tf.equal(tf.cast(logits[row, col, 0], tf.int64), tf.argmax(predictions[row, col, :cfg.num_classes])) == tf.constant(True, tf.bool):
        val += 1
  return tf.convert_to_tensor(1. * val / cfg.batch_size, tf.float32)
  
def make_onehot(inputs, num_classes):
  if np.size(np.shape(inputs)) == 1:
    y = np.zeros([np.size(inputs), num_classes])
    for row in range(np.size(inputs)):
      for col in range(num_classes):
        if inputs[row] == col:
          y[row, col] = 1.
  elif np.size(np.shape(inputs)) == 2:
    puts_debug('{}'.format([np.shape(inputs)[0], np.shape(inputs)[1], num_classes]))
    y = np.zeros([inputs.get_shape()[0], inputs.get_shape()[1], num_classes])
    for row in range(inputs.get_shape()[0]):
      for col in range(inputs.get_shape()[1]):
        for channel in range(num_classes):
          if inputs[row, col] == channel:
            y[row, col, channel] = 1.
  return tf.constant(y, tf.float32)
  
def resize(num, inputs, lines):
  length = np.size(inputs) / num
  y = np.zeros([lines, length])
  for row in range(lines):
    for col in range(length):
      if row < num:
        y[row, col] = inputs[row * length + col]
      else:
        y[row, col] = 0
  return y