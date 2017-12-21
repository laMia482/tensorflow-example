import os, sys
import cv2
import json
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from config import cfg, puts_debug, puts_info
import utility

def generate_random_data(batch_size = None, x_shape = None, y_shape = None, tensorable = True):
  if None in [batch_size, x_shape, y_shape]:
    raise ValueError('failed to generate random data cause None propoty appears')
  
  def generate_(inputs_shape = None, outputs_shape = None):
    if None in [inputs_shape]:
      raise ValueError('failed to generate cause None propoty appears')
      
    x = 10 * np.random.random(inputs_shape)
    y = int(sum(x[0, 0, :])) % 10
    
    return x, y
    
  xs, ys = [], []
  for batch_index in range(batch_size):
    x, y = generate_(inputs_shape = x_shape, outputs_shape = y_shape)
    xs.append(x)
    ys.append(y)
    
  return xs, ys

_NUM_CLASSES = 1

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer 0',
}
  
def load_tfrecord(file_name = None):
  if None in [file_name]:
    raise ValueError('None appears in fetching data, exit...')
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value = ''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value = 'jpeg'),
    'image/height': tf.FixedLenFeature([1], tf.int64),
    'image/width': tf.FixedLenFeature([1], tf.int64),
    'image/channels': tf.FixedLenFeature([1], tf.int64),
    'image/object/num': tf.FixedLenFeature((), tf.int64),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype = tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype = tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype = tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype = tf.float32),
    'image/object/class/label': tf.VarLenFeature(dtype = tf.int64)
                     }
  items_to_handles = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'height': slim.tfexample_decoder.Tensor('image/height'),
    'width': slim.tfexample_decoder.Tensor('image/width'),
    'channels': slim.tfexample_decoder.Tensor('image/channels'),
    'object/bboxes': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
    'object/labels': slim.tfexample_decoder.Tensor('image/object/class/label'),
    'object/num': slim.tfexample_decoder.Tensor('image/object/num')
                     }
  reader = tf.TFRecordReader
  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handles)
  labels_to_names = None
  # if dataset_utils.has_labels(file_name):
    # labels_to_names = dataset_utils.read_label_file(file_name)
  
  return slim.dataset.Dataset(
    data_sources = file_name,
    reader = reader,
    decoder = decoder,
    num_samples = 7807,
    items_to_descriptions = _ITEMS_TO_DESCRIPTIONS,
    num_classes = _NUM_CLASSES,
    labels_to_names = labels_to_names)
   
def tf_summary_image(image, bboxes, name = 'image', unwhitened = False):
  s_image = tf.expand_dims(image, 0)
  s_bboxes = tf.expand_dims(bboxes, 0)
  image_with_bboxes = tf.image.draw_bounding_boxes(s_image, s_bboxes)
  tf.summary.image(name, image_with_bboxes)
   
def fetch_samples(dataset = None, batch_size = None, outputs_shape = [224, 224, 3], shuffle_img = True):
  if None in [dataset, batch_size]:
    raise ValueError('None appears in fetching samples, exit...')
  provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    common_queue_capacity = 20 * batch_size,
    common_queue_min = 10 * batch_size,
    shuffle = shuffle_img
                                                            )

  batch_images, batch_num, batch_bboxes, batch_labels, batch_scores = [], [], [], [], []
  for batch_index in range(batch_size):
    image, num, bboxes, labels, height, width, channels = \
      provider.get(['image', 'object/num', 'object/bboxes', 
                    'object/labels', 'height', 'width', 'channels'
                    ])
    if [height, width] != outputs_shape[:2]:
      image = tf.image.resize_images(image, outputs_shape[:2], method = tf.image.ResizeMethod.BILINEAR, align_corners = False)
    tf_summary_image(image, bboxes)
    batch_images.append(image)
    batch_bboxes.append(bboxes)
    batch_labels.append(labels)
    batch_num.append(num)
    batch_scores.append(1.0)

  batch_images, batch_labels = tf.train.batch(
      [image, labels], 
      dynamic_pad = True, batch_size = batch_size, 
      num_threads = 4, capacity = 5 * batch_size)
  return batch_images.eval(), batch_labels.eval()
  return batch_images, batch_num, batch_bboxes, batch_labels, batch_scores
  
def load_laMia_dataset(inputs = None):
  '''
  inputs is a dataset defined by laMia, format in 
  [image, num, xmin, ymin, height, width, label, score, xmin, ymin, ...]
  1st line:         image's byte
  2nd line:         number of bounding boxes
  next num lines:   each line represent for one bounding box and its label and score
  '''
  '''
  read line by line till meeting empty line then stop
  '''
  def read_format(fp, out_type = None):
    out_str = fp.readline().strip('\n')
    out = None
    if out_str == '':
      return out_str, out, None
    if None not in [out_type]:
      out = out_type(out_str)
    return out_str, out, fp
  if None in [inputs]:
    raise ValueError('None appears in loading laMia dataset, exit...')

  provider = {}
  with open(inputs, 'r') as fp:
    index = 0
    while True:
      image_str, _, fp = read_format(fp)
      if fp is None:
        break
      num_str, num, fp = read_format(fp, out_type = int)
      if fp is None:
        break
      image_str = os.path.join('./', image_str)
      image = cv2.imread(image_str)
      if image is None:
        print('image: {} cannot open, skip...'.format(image_str))
        for i in range(num):
          fp.readline()
        continue
      image = cv2.resize(image, (224, 224)) * 1. / 255
      # image = np.random.random([224, 224, 3])
      # image = np.ones([224, 224, 3])
      bboxes, labels, scores, groundtruth = [], [], [], []
      img_info = {}
      if num > 0:
        for i in range(num):
          xmin, ymin, height, width, label, score = fp.readline().strip('\n').split(',')
          bboxes.append([float(xmin), float(ymin), float(height), float(width)])
          labels.append(int(label))
          scores.append(float(score))
          groundtruth.append([float(xmin), float(ymin), float(height), float(width), float(label), float(score)])
      img_info['image'], img_info['num'], img_info['bboxes'], img_info['labels'], img_info['scores'], img_info['groundtruth'] = \
        image, num, bboxes, labels, scores, groundtruth[0]
      provider[index] = img_info
      index += 1
  return provider

def fetch_data(provider = None, batch_size = None, outputs_shape = None, shuffle_img = True):
  if None in [provider, batch_size, outputs_shape, shuffle_img]:
    raise ValueError('None appears in fetching data, exit...')
  batch_images, batch_num, batch_bboxes, batch_labels, batch_scores, batch_gt = [], [], [], [], [], []
  examples_size = len(provider)
  for i in range(batch_size):
    if shuffle_img:
      index = np.random.randint(examples_size)
      # print('index: {}'.format(index))
    img_info = provider[index]
    # tf_image = tf.constant(img_info['image'], tf.float32)
    # tf_bboxes = tf.constant(img_info['bboxes'], tf.float32)
    # tf_summary_image(tf_image, tf_bboxes)
    batch_images.append(img_info['image'])
    batch_num.append(img_info['num'])
    batch_bboxes.append(img_info['bboxes'])
    batch_labels.append(img_info['labels'])
    batch_scores.append(img_info['scores'])
    batch_gt.append(img_info['groundtruth'])
  return np.array(batch_images), np.array(batch_gt)
  return np.array(batch_images), np.array(batch_num), np.array(batch_bboxes), np.array(batch_labels), np.array(batch_scores)

  
class Data(object):
  '''Data
  dataset is a file which written in the format of:
  {
    "1": {
      "filename": "filename",
      "width": width,
      "height": height,
      "channel": channel,
      "object_num": num,
      "object_bbox": "xmin, ymin, width, height" * num (value normalized to [0, 1]),
      "object_label": "0, 1, 2, ..." (size is num)
    },
    "2": {
      "filename": "filename",
      ...
    }...
  }
  '''
  def __init__(self):
    self._provider = []
    self._groundtruth = []
    self._size = 0
    
  def load(self, data_filename = None):
    if None in [data_filename]:
      raise ValueError('data is empty')
    with open(data_filename) as fp:
      ctx = json.load(fp)
      fp.close()
      for key in ctx.keys():
        record, prop = ctx[key], {}
        image = cv2.imread(record['filename'])
        if image is None:
          continue
        # resize image to be continue
        num = record['object_num']
        labels = [float(val) for val in record['object_label'].split(',')]
        labels = utility.reshape(int(num), labels, cfg.max_predictions)
        bboxes = [float(val) for val in record['object_bbox'].split(',')]
        bboxes = utility.reshape(int(num), bboxes, cfg.max_predictions)
        puts_debug('{}\nval: \n{}, shape: {}\nval: \n{}, shape: {}'.format(num, labels, np.shape(labels), bboxes, np.shape(bboxes)))
        prop['raw_data'], prop['num'], prop['labels'], prop['bboxes'] = image / 255., int(num), labels, bboxes
        self._provider.append(prop)
        prop = np.concatenate([prop['labels'], prop['bboxes']], axis = 1)
        puts_debug('prop shape: {}'.format(np.shape(prop)))
        self._groundtruth.append(prop)
        self._size += 1
    
  def decode_and_fetch(self, batch_size = None):
    raw_data, labels, bboxes, groundtruth = [], [], [], []
    for i in range(batch_size):
      index = np.random.randint(0, self._size)
      raw_data.append(self._provider[index]['raw_data'])
      labels.append(self._provider[index]['labels'])
      bboxes.append(self._provider[index]['bboxes'])
      groundtruth.append(self._groundtruth[index])
    puts_debug('{}, {}, {}, {}'.format(np.shape(raw_data), np.shape(labels), np.shape(bboxes), np.shape(groundtruth)))
    return raw_data, labels, bboxes, groundtruth
    return tf.convert_to_tensor(raw_data, tf.float32), tf.convert_to_tensor(labels, tf.float32), tf.convert_to_tensor(bboxes, tf.float32), tf.convert_to_tensor(groundtruth, tf.float32)
    
  def size(self):
    return self._size
