import os
import json
import cv2
import numpy as np

work_dir = 'train'
output = 'out.json'

def main(prefix, dst):
  categories = ['printed_face', 'real_face']
  cate2num = {'printed_face': 0, 'real_face': 1}
  prop = {}
  index = 0
  for category in categories:
    for filename in sorted(os.listdir(os.path.join(prefix, category))):
      image_path = os.path.join(prefix, category, filename)
      image = cv2.imread(image_path)
      if image is None:
        continue
      prop[index] = {}
      prop[index]['filename'] = os.path.abspath(image_path)
      prop[index]['width'] = np.shape(image)[0]
      prop[index]['height'] = np.shape(image)[1]
      prop[index]['channel'] = np.shape(image)[2]
      prop[index]['object_num'] = 1
      prop[index]['object_labels'] = []
      prop[index]['object_bboxes'] = []
      for i in range(prop[index]['object_num']):
        prop[index]['object_labels'].append(cate2num[category])
        prop[index]['object_bboxes'] += [0.0, 0.0, prop[index]['width'] / prop[index]['width'], prop[index]['height'] / prop[index]['height']]
      index += 1
      
  with open(dst, 'w') as fp:
    json.dump(prop, fp, ensure_ascii = False)
    fp.close()

if __name__ == '__main__':
  main(work_dir, output)
