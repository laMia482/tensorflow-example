import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from config import puts_debug, puts_info
  
def build(network_name):
  if network_name in network_map:
    return network_map[network_name]

def default_network(inputs = None, max_predictions = None, num_classes = None, is_train = None, scope = 'default_network'):
  if None in [inputs, max_predictions, num_classes, is_train]:
    raise ValueError('None should never arise in default_network')
  with tf.variable_scope(scope, 'network', [inputs]):
    end_points = {}
    end_points['inputs'] = inputs
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                                      activation_fn = tf.nn.relu, 
                                      # weights_initializer = tf.truncated_normal_initializer(0, 1),
                                      # weights_regularizer = slim.l2_regularizer(.01),
                                      ):
      net = slim.conv2d(net,  64, [5, 5], stride = 2, padding = 'VALID', scope = 'conv1')
      net = slim.conv2d(net, 128, [3, 3], stride = 2, padding = 'VALID', scope = 'conv2')
      net = slim.conv2d(net, 256, [3, 3], stride = 2, padding = 'VALID', scope = 'conv3')
      net = slim.conv2d(net, 512, [3, 3], stride = 2, padding = 'VALID', scope = 'conv4')
      net = tf.reshape(net, [-1, 2 * 2 * 512])
      end_points['reshape'] = net
      
      net = slim.stack(end_points['reshape'], slim.fully_connected, [1024, 1024, 1024, 512, max_predictions * num_classes], scope = 'fc4_stack_class')
      end_points['fc4_stack_class'] = net
      net = tf.reshape(net, [-1, max_predictions, num_classes])
      end_points['fc4_stack_class_reshape'] = net
      
      net = slim.stack(end_points['reshape'], slim.fully_connected, [1024, 1024, 1024, 512, max_predictions * 4], scope = 'fc4_stack_bboxes')
      end_points['fc4_stack_bboxes'] = net
      net = tf.reshape(net, [-1, max_predictions, 4])
      end_points['fc4_stack_bboxes_reshape'] = net
      
      net = tf.concat([end_points['fc4_stack_class_reshape'], end_points['fc4_stack_bboxes_reshape']], 2)
      end_points['outputs'] = net
      puts_debug('network output shape: {}'.format(net.get_shape()))
      return net, end_points
      
network_map = {
  'default_network': default_network,
              }