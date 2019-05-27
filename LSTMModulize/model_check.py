import tensorflow as tf
import numpy as np


tf.reset_default_graph()
sess = tf.Session()

new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
new_saver.restore(sess, 'model/model.ckpt')

graph = tf.get_default_graph()
name = [n.name for n in tf.trainable_variables()]
for n in name:
    print_value = graph.get_tensor_by_name(n)
    print(n, sess.run(print_value))