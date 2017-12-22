import os
import tensorflow as tf
from config import cfg, puts_debug, puts_info
import data_reader as dtrd
import network
import utility

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
run_net = network.build(cfg.network)

xs = tf.placeholder(tf.float32, [cfg.batch_size, cfg.image_width, cfg.image_height, cfg.image_channel])
ys = tf.placeholder(tf.float32, [cfg.batch_size, cfg.max_predictions, 5])
learning_rate = tf.placeholder(tf.float32, None)
outputs_op, _ = run_net(inputs = xs, max_predictions = cfg.max_predictions, num_classes = cfg.num_classes, is_train = cfg.is_train)
loss_op = utility.calc_loss(logits = ys, predictions = outputs_op)
train_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(100 * loss_op)
accuracy_op = utility.calc_accuracy(logits = ys, predictions = outputs_op)

def train():
  '''kernel train
  
  '''
  with tf.Session(config = config) as sess:
    saver = tf.train.Saver()
    global_step = 0
    iter= 1
    current_learning_rate = cfg.init_learning_rate
    init_variable = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_variable)
    data = dtrd.Data()
    data.load(data_filename = cfg.train_dataset)
    puts_debug('data size: {}'.format(data.size()))
    while global_step < data.size() * cfg.epoch:
      batch_x, _, _, batch_y = data.decode_and_fetch(batch_size = cfg.batch_size)
      _, loss_val = sess.run([train_op, loss_op], feed_dict = {xs: batch_x, ys: batch_y, learning_rate: current_learning_rate})
      puts_info('iter: {}, loss: {}'.format(iter, loss_val))
      global_step += cfg.batch_size
      
      if iter % cfg.test_iter == 0:
        batch_x, _, _, batch_y = data.decode_and_fetch(batch_size = cfg.batch_size)
        accuracy_val, loss_val = sess.run([accuracy_op, loss_op], feed_dict = {xs: batch_x, ys: batch_y, learning_rate: current_learning_rate})
        puts_info('accuracy: {:.4f}, loss: {:.4f}'.format(accuracy_val, loss_val))
        
      if iter % cfg.save_iter == 0:
        saver.save(sess, os.path.join(cfg.save_path, 'model.ckpt-' + str(global_step)))
        puts_info('iter: {}, model has been saved under {}/model.ckpt-{}'.format(iter, cfg.save_path, global_step))

      iter += 1
      
    batch_x, _, _, batch_y = data.decode_and_fetch(batch_size = cfg.batch_size)
    accuracy_val, loss_val, outputs_val = sess.run([accuracy_op, loss_op, outputs_op], feed_dict = {xs: batch_x, ys: batch_y, learning_rate: current_learning_rate})
    puts_info('final >> val: \n{}, accuracy: {:.4f}, loss: {:.4f}'.format(outputs_val, accuracy_val, loss_val))
    saver.save(sess, os.path.join(cfg.save_path, 'model.ckpt'))
    puts_info('final model has been saved under {}'.format(os.path.join(cfg.save_path, 'model.ckpt')))
  
  
  
def eval():
  '''kernel eval
  
  '''
  return