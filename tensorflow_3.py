print('variable 실습')

import tensorflow as tf

_x = tf.placeholder(tf.float32, shape=[])
w = tf.get_variable('weight', [])
three = tf.constant(3.0)
mul = tf.multiply(_x, w)
add = tf.add(mul, three)

sess = tf.Session()
init_op = w.assign(2.0)
#init_op = tf.variables_initializer([w])
sess.run(init_op)
y = sess.run(add, feed_dict={_x:10})
print(y)
