import numpy as np
import tensorflow as tf
from tf.compat.v1.nn.static_rnn import rnn

data=np.random.randint(0,10,[1000,10,2])
input_x=tf.placeholder(tf.float32,[1000,10,2])

with tf.variable_scope('encoder') as scope:
    cell=rnn.LSTMCell(150)
    model=tf.nn.dynamic_rnn(cell,inputs=input_x,dtype=tf.float32)

    output_,(fs,fc)=model



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(model, feed_dict={input_x: data})

    print(output)
