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
