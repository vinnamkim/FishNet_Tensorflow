import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('batch_norm_decay', 0.997, 'to test')
flags.DEFINE_float('batch_norm_epsilon', 1e-5, 'to test')
flags.DEFINE_bool('batch_norm_fused', True, 'to test')