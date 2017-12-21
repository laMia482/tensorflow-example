import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from config import puts_debug, puts_info
  
def build(network_name):
  if network_name in network_map:
    return network_map[network_name]
    
def default_network(inputs = None, num_classes = None, is_training = None, scope = 'default_network', inputs_shape = [1, 1, 3]):
  if inputs.get_shape()[1:] != inputs_shape:
    raise ValueError('demension not match feeding inputs')
  with tf.variable_scope(scope, 'net', [inputs]):
    end_points = {}
    with slim.arg_scope([slim.fully_connected], activation_fn = tf.nn.relu, ):
      end_points['inputs'] = inputs
      net = inputs
      # net = tf.squeeze(net, [1, 2])
      net = slim.repeat(net, 3, slim.max_pool2d, 2, scope = 'pool1') # 224 -> 112 -> 56 -> 28
      net = tf.reshape(net, [-1, 28 * 28 * 3])
      net = slim.stack(net, slim.fully_connected, [1024, 64, 128, 256, 512, 1024])
      net = slim.fully_connected(net, num_classes, activation_fn = tf.nn.sigmoid, scope = 'outputs')
      end_points['outputs'] = net
      return net
      
def tf_swish(x):
  return tf.nn.relu(x) * tf.nn.sigmoid(x)

def laMia_network(inputs = None, num_classes = None, is_training = None, scope = 'laMia_network', inputs_shape = [1, 1, 3]):
  if inputs.get_shape()[1:] != inputs_shape:
    raise ValueError('demension not match feeding inputs')
  with tf.variable_scope(scope, 'network', [inputs]):
    end_points = {}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = tf.nn.sigmoid,
                        weights_initializer = tf.truncated_normal_initializer(0, .01),
                        weights_regularizer = slim.l2_regularizer(.01),
    ):
      end_points['inputs'] = inputs
      with slim.arg_scope([slim.conv2d], activation_fn = tf_swish, stride = 1, padding = 'VALID'):
        net = slim.conv2d(inputs, 32, [5, 5], scope = 'conv1')
        net = slim.max_pool2d(net, 2, scope = 'pool1')
        sc_net = slim.max_pool2d(net, 2, scope = 'sc_pool1')
        sc_net = tf.reshape(sc_net, [-1, 55 * 55 * 32])
        sc_net = slim.stack(sc_net, slim.fully_connected, [512], scope = 'sc_net')
        net = slim.repeat(net, 5, slim.conv2d, 64, [3, 3], scope = 'conv2')
        net = slim.max_pool2d(net, 2, scope = 'pool2')
        net = slim.dropout(net, .8, is_training = is_training)
        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope = 'conv3')
        net = slim.max_pool2d(net, 2, scope = 'pool3')
        net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope = 'conv4')
        net = slim.max_pool2d(net, 2, scope = 'pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope = 'conv5')
        # net = slim.conv2d(net, 6, [1, 1], scope = 'conv6')
        # net = tf.squeeze(net, [1, 2])
        net = tf.reshape(net, [-1, 1 * 1 * 512])
        net = tf.concat([net, sc_net], 1)
        net = slim.fully_connected(net, 1024, scope = 'fc6')
        net = slim.fully_connected(net, num_classes + 5, scope = 'fc7')
        end_points['out'] = net
        return net

def miy_network(inputs = None, num_classes = None, is_training = None, scope = 'conv_network', inputs_shape = [1, 1, 3]):
  if inputs.get_shape()[1:] != inputs_shape:
    raise ValueError('demension not match feeding inputs')
  with tf.variable_scope(scope, 'network', [inputs]):
    end_points = {}
    end_points['inputs'] = inputs
    net = inputs # 224
    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                        activation_fn = tf.nn.relu,
                        # weights_initializer = tf.truncated_normal_initializer(0, .01),
                        # weights_regularizer = slim.l2_regularizer(.01),
                        ):
      net = slim.repeat(net, 3, slim.max_pool2d, 2, scope = 'pool1') # 224 -> 112 -> 56 -> 28
      # net = tf.reshape(net, [-1, 14 * 14 * 3])
      net = slim.conv2d(net, 512, [11, 11], stride = 2, padding = 'VALID', scope = 'conv1')
      net = slim.conv2d(net, 512, [9, 9], padding = 'VALID', scope = 'conv2')
      net = tf.squeeze(net, [1, 2])
      net = slim.stack(net, slim.fully_connected, [512, 512, 1024, 4096, 1024], scope = 'fc1')
      net = slim.fully_connected(net, num_classes + 5, activation_fn = tf.nn.sigmoid, scope = 'fc2')
      end_points['out'] = net
      return net

def repo_network(inputs = None, max_predictions = None, num_classes = None, is_train = None, scope = 'repo_network'):
  if None in [inputs, max_predictions, num_classes, is_train]:
    raise ValueError('None should never arise in repo_network')
  with tf.variable_scope(scope, 'network', [inputs]):
    end_points = {}
    end_points['inputs'] = inputs
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                                      activation_fn = tf.nn.relu, 
                                      ):
      net = slim.conv2d(net, 128, [15, 15], stride = 10, padding = 'VALID', scope = 'conv1')
      net = slim.conv2d(net, 128, [15, 15], stride = 1, padding = 'VALID', scope = 'conv2')
      net = tf.reshape(net, [-1, 7 * 7 * 128])
      net = slim.stack(net, slim.fully_connected, [1024, 1024, 1024, 512, max_predictions * (num_classes + 4)], scope = 'fc1_stack')
      net = tf.reshape(net, [-1, max_predictions, num_classes + 4])
      end_points['outputs'] = net
      puts_debug('network output shape: {}'.format(net.get_shape()))
      return net, end_points
      
network_map = {
  'default_network': default_network,
  'laMia_network': laMia_network,
  'miy_network': miy_network,
  'repo_network': repo_network,
              }