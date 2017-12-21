'''
import os, sys
import time
import math
import numpy as np
import cv2
import tensorflow as tf
import data_reader as reader
import network as net
import utility

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('is_eval', False, 'if need to eval model')
flags.DEFINE_string('is_training', True, 'if need to train model')
flags.DEFINE_string('print_model', True, 'if need to print model')
flags.DEFINE_string('network', 'default_network', 'network')
flags.DEFINE_integer('num_classes', 6, 'number of classes')
flags.DEFINE_integer('max_object_propals', 1, 'propals objects')
flags.DEFINE_integer('batch_size', 32, 'default batch size')
flags.DEFINE_string('inputs', './train.record', 'inputs dataset')
flags.DEFINE_string('outputs_dir', 'ckpt', 'dir to save ckpt files')
flags.DEFINE_string('optimizer', 'default_optimizer', 'default optimizer')
flags.DEFINE_integer('max_batches', 10000 * FLAGS.batch_size, 'max batches for training')
flags.DEFINE_float('init_learning_rate', 0.1, 'initialize learning rate')


inputs_shape = [224, 224, 3]
dataset = reader.load_tfrecord(FLAGS.inputs)

def train_model(network = None, num_classes = None, 
                batch_size = None, inputs = None, outputs_dir = None, 
                optimizer_method = None):
  if None in [network, num_classes, batch_size, inputs, outputs_dir, optimizer_method]:
    raise ValueError('None appears in training model, exit...')
  provider = reader.load_laMia_dataset(inputs = inputs)
  predictor = net.build(FLAGS.network)
  xs = tf.placeholder(tf.float32, [None] + inputs_shape)
  ys = tf.placeholder(tf.float32, [None, 6])
  learning_rate = tf.placeholder(tf.float32, None)
  index = tf.placeholder(tf.int32, None)
  # outputs should be [xmin, ymin, width, height, label, score]
  outputs = predictor(inputs = xs, num_classes = num_classes, is_training = FLAGS.is_training, inputs_shape = inputs_shape)
  # ys_onehot_labels = tf.one_hot(indices = ys, depth = num_classes)
  # print('y: {}, ys: {}, out: {}'.format(np.shape(ys), ys_onehot_labels.get_shape(), outputs.get_shape()))
  outputs_labels = tf.argmax(outputs[:, 4:num_classes + 4], 1)
  labels_loss = tf.losses.softmax_cross_entropy(onehot_labels = ys[:, 4], logits = tf.cast(outputs_labels, tf.float32))
  coords_loss = tf.losses.mean_squared_error(labels = ys[:, :4], predictions = outputs[:, :4])
  loss = labels_loss + coords_loss
  train = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
  # train = tf.train.AdamOptimizer(learning_rate = FLAGS.init_learning_rate).minimize(loss)
  # accuracy = tf.metrics.accuracy(labels = tf.cast(.05 + ys[:, 4], tf.int32), predictions = tf.cast(.05 + outputs[:, 4], tf.int32))[1]
  # accuracy = tf.metrics.accuracy(labels = tf.argmin(ys), predictions = tf.argmin(outputs))[1]
  iou = tf.metrics.mean_iou(labels = ys[:, :4], predictions = outputs[:, :4], num_classes = 1)[0]
  accuracy = utility.calc_accuracy(ys[:, 4], outputs_labels, batch_size)
  # iou = calc_iou(ys[:, :4], outputs[:, :4])
  # score = calc_score(ys[:, 5], outputs[:, num_classes + 4])

  # def calc_iou(x, y):


  data_info = tf.concat([ys, outputs], 1)

  with tf.Session() as sess:
    saver = tf.train.Saver()
    batch_index = 0
    batch_learning_rate = FLAGS.init_learning_rate
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    while batch_index < FLAGS.max_batches:
      # batch_x, batch_y = reader.fetch_samples(dataset = dataset, batch_size = batch_size, outputs_shape = inputs_shape)
      # batch_x, batch_y = reader.generate_random_data(batch_size = batch_size, x_shape = [1, 1, 3], y_shape = [num_classes])
      batch_x, batch_y = reader.fetch_data(provider = provider, batch_size = batch_size, outputs_shape = inputs_shape)
      # print('batch_x: {}, {}, batch_y: {}, {}'.format(np.shape(batch_x), type(batch_x), np.shape(batch_y), type(batch_y)))
      if batch_learning_rate > 0.0001:
        batch_learning_rate = FLAGS.init_learning_rate * math.pow(0.1, batch_index / 2000 / batch_size)
      _, loss_val = sess.run([train, loss], feed_dict = {xs: batch_x, ys: batch_y, learning_rate: batch_learning_rate, index: batch_index})
      # xs, ys = batch_x, batch_y
      # _, loss_val = sess.run([train, loss])
      batch_index += batch_size
      
      if batch_index % (1 * batch_size) == 0:
        # test_x, test_y = reader.fetch_samples(dataset = dataset, batch_size = batch_size, outputs_shape = inputs_shape)
        # test_x, test_y = reader.generate_random_data(batch_size = batch_size, x_shape = [1, 1, 3], y_shape = [num_classes])
        test_x, test_y = reader.fetch_data(provider = provider, batch_size = batch_size, outputs_shape = inputs_shape)
        start_time = time.time()
        accuracy_val, iou_val, outputs_val, data_info_val = sess.run([accuracy, iou, outputs, data_info], feed_dict = {xs: test_x, ys: test_y})
        stop_time = time.time()
        # print('[{}, {}, {}, {}, {}, {}]'.format(test_y[0, 0], test_y[0, 1], test_y[0, 2], test_y[0, 3], test_y[0, 4], test_y[0, 5]))
        print('train: batch_index: {:8d}, rate: {:6f}, accuracy: {:.6f}, iou: {:.6f} loss: {:.6f}, time: {:.6f} seconds'.format(batch_index, batch_learning_rate, accuracy_val, iou_val, loss_val, stop_time - start_time))
        if batch_index % (50 * batch_size) == 0:
          print('outputs: \n{}'.format(outputs_val))
      # if batch_index % (10 * 32) == 0:
    saver.save(sess, os.path.join(FLAGS.outputs_dir, 'model.ckpt-') + str(batch_index))
  
  
def eval_model(network = None, num_classes = None, inputs = None):
  if None in [network, num_classes, inputs]:
    raise ValueError('None appears in evaling model, exit...')
  with tf.Session() as sess:
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    saver = tf.train.import_meta_graph(tf.train.latest_checkpoint('./ckpt') + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))
    # saver.restore(sess, 'ckpt/model.ckpt-320000')
    provider = reader.load_laMia_dataset(inputs = inputs)
    # test_x, test_y = reader.fetch_samples(dataset = dataset, batch_size = batch_size, outputs_shape = inputs_shape)
    # test_x, test_y = reader.generate_random_data(batch_size = FLAGS.batch_size, x_shape = [1, 1, 3], y_shape = [num_classes])
    test_x, test_y = reader.fetch_data(provider = provider, batch_size = FLAGS.batch_size, outputs_shape = inputs_shape)
    outputs_tensor = tf.get_default_graph().get_tensor_by_name('conv_network/fc2:0')
    start_time = time.time()
    accuracy_val, outputs_val = sess.run([accuracy, outputs_tensor], feed_dict = {xs: test_x, ys: test_y})
    stop_time = time.time()
    print('eval: batch_index: {:8d}, accuracy: {:.6f}, iou: {:.6f} loss: {:.6f}, time: {:.6f} seconds'.format(batch_index, accuracy_val, iou_val, loss_val, stop_time - start_time))
  
def print_model(ckpt_dir = None):
  from tensorflow.python import pywrap_tensorflow  
  # checkpoint_path = os.path.join(ckpt_dir, "model.ckpt-320000")  
  checkpoint_path = tf.train.latest_checkpoint('./ckpt')
  reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
  var_to_shape_map = reader.get_variable_to_shape_map()  
  for key in var_to_shape_map:  
    print("tensor_name: {}".format(key))
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names  


def main(_):
  if FLAGS.is_training:
    train_model(network = FLAGS.network, num_classes = FLAGS.num_classes, 
                batch_size = FLAGS.batch_size, 
                inputs = FLAGS.inputs, outputs_dir = FLAGS.outputs_dir,
                optimizer_method = FLAGS.optimizer)
  if FLAGS.print_model:
    print_model(FLAGS.outputs_dir)
  if FLAGS.is_eval:
    eval_model(network = FLAGS.network, num_classes = FLAGS.num_classes,
               inputs = FLAGS.inputs, )
               
if __name__ == '__main__':
  tf.app.run()
  
  
'''


from config import cfg
import kernel
import tensorflow as tf

def main(_):
  if cfg.is_train is True:
    kernel.train()
  if cfg.is_eval is True:
    kernel.eval()

if __name__ == '__main__':
  tf.app.run()
