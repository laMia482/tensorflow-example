import os, sys
import cv2
import json
import numpy as np
from config import cfg, puts_debug, puts_info
import utility

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
      "object_bbox": "xmin, ymin, xmax, ymax" * num (value normalized to [0, 1]),
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
        labels = [float(val) for val in record['object_labels']]
        labels = utility.resize(int(num), labels, cfg.max_predictions)
        bboxes = [float(val) for val in record['object_bboxes']]
        bboxes = utility.resize(int(num), bboxes, cfg.max_predictions)
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
    # return tf.convert_to_tensor(raw_data, tf.float32), tf.convert_to_tensor(labels, tf.float32), tf.convert_to_tensor(bboxes, tf.float32), tf.convert_to_tensor(groundtruth, tf.float32)
    
  def size(self):
    return self._size
